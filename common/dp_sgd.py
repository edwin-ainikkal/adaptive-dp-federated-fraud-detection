"""
common/dp_sgd.py
------------------
Local (client-side) training with ADAPTIVE per-sample DP-SGD.

One call of LocalDPTrainer.train_round() = the "local training" phase of one
federated round. For every mini-batch step:

  1. Poisson-sample a batch (each record is included independently with
     probability q = batch_size / n_samples)  -> needed for the accountant.
  2. Compute each sample's gradient norm per layer tensor (W1,b1,...,b3).
  3. CLIP each sample's gradient to that layer's threshold C_j
        adaptive: C_j tracked by AdaptiveClipper (EMA)
        fixed   : C_j = 1.0
  4. SUM the clipped gradients, add Gaussian noise  N(0, (z_j * C_j)^2)  per
     layer, divide by the expected batch size.
        adaptive: z_j = z / sqrt(share_j)  -- the per-step privacy budget is
                  split across layers by Fisher-style importance (important
                  layers -> bigger share -> LESS noise).
        fixed   : share_j = 1/L (equal split).
     Both modes spend the SAME nominal budget (overall multiplier z).
  5. Take an Adam step on the noisy gradient (post-processing).
  6. Charge the Renyi-DP accountant. A "privacy filter" stops local training
     before the epsilon budget would be exceeded.

dp=False runs ordinary mini-batch training (no clipping/noise) = the
"no privacy" reference run.
"""

import numpy as np

from common.adaptive_dp import AdaptiveClipper, FisherImportance, RDPAccountant
from common.model import AdamOptimizer, LAYER_NAMES

WEIGHT_BIAS = [("W1", "b1"), ("W2", "b2"), ("W3", "b3")]


class LocalDPTrainer:
    def __init__(self, n_samples, batch_size=256, epochs=2, lr=0.01, pos_weight=5.0,
                 noise_multiplier=1.0, adaptive=True, dp=True, delta=1e-5,
                 epsilon_budget=None, fixed_clip=1.0, clip_quantile=0.5, seed=0):
        self.N = n_samples
        self.B = batch_size
        self.q = min(1.0, batch_size / n_samples)
        self.epochs, self.lr, self.pos_weight = epochs, lr, pos_weight
        self.z, self.adaptive, self.dp = noise_multiplier, adaptive, dp
        self.fixed_clip = fixed_clip
        self.epsilon_budget = epsilon_budget
        self.rng = np.random.default_rng(seed)
        self.clipper = AdaptiveClipper(LAYER_NAMES, init_clip=fixed_clip, target_quantile=clip_quantile)
        self.fisher = FisherImportance(LAYER_NAMES)
        self.accountant = RDPAccountant(delta)
        self.sigma_b = batch_size / 20.0       # noise on the clip-tracking counts (Andrew et al.)
        self.exhausted = False

    # -- helpers -----------------------------------------------------------
    def _norms(self, inputs, dzs):
        norms = {}
        for l, (wn, bn) in enumerate(WEIGHT_BIAS):
            dz_n = np.linalg.norm(dzs[l], axis=1)
            norms[wn] = np.linalg.norm(inputs[l], axis=1) * dz_n
            norms[bn] = dz_n
        return norms

    def _round_sigmas(self):
        L = len(LAYER_NAMES)
        shares = self.fisher.shares() if self.adaptive else np.full(L, 1.0 / L)
        z_layer = self.z / np.sqrt(shares)
        mechs = list(z_layer) + ([self.sigma_b] * L if self.adaptive else [])
        return shares, z_layer, mechs

    # -- one federated round of local training ------------------------------
    def train_round(self, model, X, y):
        steps_per_epoch = max(1, int(round(1.0 / self.q)))
        shares, z_layer, mechs = self._round_sigmas()
        opt = AdamOptimizer(model.params, lr=self.lr)
        losses, epochs_done = [], 0

        for ep in range(self.epochs):
            if self.dp and self.epsilon_budget is not None:          # PRIVACY FILTER
                trial = self.accountant.copy()
                trial.add_steps(self.q, mechs, steps_per_epoch)
                if trial.epsilon() > self.epsilon_budget:
                    self.exhausted = True
                    print(f"    [privacy filter] budget eps={self.epsilon_budget} would be exceeded "
                          f"-> local training stopped.")
                    break
            ep_loss, n_ep = 0.0, 0
            for _ in range(steps_per_epoch):
                idx = np.flatnonzero(self.rng.random(self.N) < self.q)
                if idx.size == 0:
                    continue
                xb, yb = X[idx], y[idx]
                _, cache = model.forward(xb)
                (inputs, dzs), loss = model.per_sample_backward(cache, yb, self.pos_weight)
                grads = self._dp_step_gradients(inputs, dzs, z_layer, len(idx))
                opt.step(model.params, grads)
                ep_loss += loss; n_ep += 1
            if self.dp:
                self.accountant.add_steps(self.q, mechs, steps_per_epoch)
            epochs_done += 1
            losses.append(ep_loss / max(n_ep, 1))
            eps_txt = f" | eps={self.epsilon():.3f}" if self.dp else ""
            print(f"    [client] local epoch {ep + 1}/{self.epochs} | avg loss={losses[-1]:.5f}{eps_txt}")

        return {"epsilon": self.epsilon(), "exhausted": self.exhausted, "epochs_done": epochs_done,
                "clip": dict(self.clipper.clip), "shares": [float(s) for s in shares]}

    def epsilon(self):
        return self.accountant.epsilon() if self.dp else float("nan")

    def _dp_step_gradients(self, inputs, dzs, z_layer, batch_n):
        norms = self._norms(inputs, dzs)
        grads = {}
        for l, (wn, bn) in enumerate(WEIGHT_BIAS):
            for name in (wn, bn):
                if self.dp:
                    C = self.clipper.clip[name] if self.adaptive else self.fixed_clip
                    f = np.minimum(1.0, C / (norms[name] + 1e-12))
                else:
                    f = np.ones_like(norms[name])
                if name == wn:
                    g = (inputs[l] * f[:, None]).T @ dzs[l]
                else:
                    g = (dzs[l] * f[:, None]).sum(axis=0)
                if self.dp:
                    j = LAYER_NAMES.index(name)
                    sigma = z_layer[j] * C
                    g = (g + self.rng.normal(0.0, sigma, size=g.shape)) / self.B
                    if self.adaptive:
                        below = float(np.sum(norms[name] <= C))
                        frac = (below + self.rng.normal(0.0, self.sigma_b)) / self.B
                        self.clipper.update(name, float(np.clip(frac, 0.0, 1.0)))
                        self.fisher.update(name, g, sigma / self.B)
                else:
                    g = g / batch_n
                grads[name] = g.astype(np.float32)
        return grads
