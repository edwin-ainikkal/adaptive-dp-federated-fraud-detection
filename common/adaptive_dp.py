"""
common/adaptive_dp.py
-----------------------
The building blocks of the ADAPTIVE differential-privacy mechanism.

The training loop that uses them is common/dp_sgd.py (per-sample DP-SGD on
each client). This file contains the three ideas from the project's
methodology plus the privacy accountant:

 1. AdaptiveClipper    "dynamic gradient clipping ... exponential moving
                        average (EMA) of historical gradient patterns".
 2. FisherImportance   "selective noise injection using layer-wise Fisher
                        information ... less noise on important parameters".
 3. RDPAccountant      "privacy filter ... track realized privacy loss in
                        real time" (Renyi-DP accountant for the sub-sampled
                        Gaussian mechanism, Mironov et al. 2019).

PRIVACY NOTES (useful for the viva)
 * Guarantee type: RECORD-level (every individual transaction) (eps, delta)-DP
   for each bank's data, via DP-SGD with per-sample clipping + Gaussian noise.
 * The clip thresholds are tracked from a PRIVATELY NOISED statistic (a noisy
   count of how many per-sample gradient norms fall below the clip -- Andrew
   et al., "Differentially Private Learning with Adaptive Clipping", 2021),
   smoothed with an EMA. The EMA is post-processing, so it costs no extra
   privacy; the noisy count itself is charged to the accountant.
 * The Fisher-style importance is computed ONLY from the already-noised
   gradients (with the known noise variance subtracted), so it is
   post-processing and also free. It is applied from the next FL round on.
"""

import numpy as np
from scipy.special import gammaln, logsumexp

DEFAULT_ORDERS = list(range(2, 65)) + [80, 96, 128, 192, 256]


# ----------------------------------------------------------------------------
# 1. Adaptive clipping (EMA of a privately-estimated quantile signal)
# ----------------------------------------------------------------------------
class AdaptiveClipper:
    """Keeps one clip threshold C per layer tensor and moves it so that a
    target fraction (default: the median) of per-sample gradient norms
    falls below it.

        frac_ema  = ema*frac_ema + (1-ema)*noisy_fraction_below_clip   (EMA)
        C         = C * exp( -lr * (frac_ema - target_quantile) )

    If too few samples are below C (frac < target) the clip grows; if too many,
    it shrinks. So C follows the *historical* gradient-norm scale instead of
    being one fixed number for the whole run."""

    def __init__(self, names, init_clip=1.0, target_quantile=0.5, lr=0.2, ema=0.9,
                 min_clip=1e-4, max_clip=1e3):
        self.names = list(names)
        self.target = target_quantile
        self.lr, self.ema = lr, ema
        self.min_clip, self.max_clip = min_clip, max_clip
        self.clip = {n: float(init_clip) for n in self.names}
        self.frac_ema = {n: float(target_quantile) for n in self.names}

    def update(self, name, noisy_fraction_below):
        self.frac_ema[name] = self.ema * self.frac_ema[name] + (1 - self.ema) * noisy_fraction_below
        c = self.clip[name] * np.exp(-self.lr * (self.frac_ema[name] - self.target))
        self.clip[name] = float(np.clip(c, self.min_clip, self.max_clip))


# ----------------------------------------------------------------------------
# 2. Fisher-style layer importance -> privacy-budget shares
# ----------------------------------------------------------------------------
class FisherImportance:
    """Layer-wise (diagonal-)Fisher-style importance.

    The Fisher information of a layer is the expected squared gradient, so
    sqrt(d * mean(g^2)) is the (square-root of the) Fisher trace = the layer's
    gradient norm. We estimate it from the NOISY gradient the optimiser
    actually used, subtracting the known noise power (so the estimate is
    unbiased) and smoothing with an EMA. Pure post-processing => no privacy cost.

    shares() converts importances into budget shares that sum to 1. A layer
    with a larger share of the per-step privacy budget gets LESS noise."""

    def __init__(self, names, floor=0.2, ema=0.95):
        self.names = list(names)
        self.floor, self.ema = floor, ema
        self.value = {n: 0.0 for n in self.names}

    def update(self, name, noisy_mean_grad, noise_std_on_mean):
        d = noisy_mean_grad.size
        ms = float(np.mean(noisy_mean_grad ** 2)) - noise_std_on_mean ** 2
        est = float(np.sqrt(max(ms, 0.0) * d))
        self.value[name] = self.ema * self.value[name] + (1 - self.ema) * est

    def shares(self):
        v = np.array([self.value[n] for n in self.names])
        if v.max() <= 0:
            return np.full(len(v), 1.0 / len(v))
        imp = self.floor + (1 - self.floor) * (v / v.max())   # in [floor, 1]
        return imp / imp.sum()


# ----------------------------------------------------------------------------
# 3. Renyi-DP accountant (privacy filter)
# ----------------------------------------------------------------------------
def _rdp_sampled_gaussian(q, sigma, orders):
    """RDP of the Poisson-sub-sampled Gaussian mechanism at integer orders
    (Mironov, Talwar & Zhang 2019). q = sampling rate, sigma = noise
    multiplier (noise std / sensitivity)."""
    out = []
    for a in orders:
        k = np.arange(a + 1)
        log_terms = (gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1)
                     + (a - k) * np.log1p(-q) + k * np.log(q)
                     + (k * k - k) / (2.0 * sigma ** 2))
        out.append(logsumexp(log_terms) / (a - 1))
    return np.array(out)


class RDPAccountant:
    def __init__(self, delta=1e-5, orders=DEFAULT_ORDERS):
        self.delta = delta
        self.orders = np.array(orders, dtype=float)
        self.rdp = np.zeros(len(orders))

    def copy(self):
        c = RDPAccountant(self.delta, self.orders.astype(int).tolist())
        c.rdp = self.rdp.copy()
        return c

    def add_steps(self, q, sigmas, steps):
        """`sigmas` = noise multipliers of EVERY Gaussian mechanism released in
        one step (one per layer, plus the clip-tracking counts if adaptive).
        RDP composes by addition over mechanisms and over steps."""
        if steps <= 0:
            return
        per_step = sum(_rdp_sampled_gaussian(q, s, self.orders.astype(int)) for s in sigmas)
        self.rdp = self.rdp + steps * per_step

    def epsilon(self):
        a = self.orders
        eps = (self.rdp - (np.log(self.delta) + np.log(a)) / (a - 1)
               + np.log((a - 1) / a))                      # Balle et al. 2020 conversion
        return float(max(np.min(eps), 0.0))
