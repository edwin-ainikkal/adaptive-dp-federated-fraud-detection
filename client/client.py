"""
client/client.py
------------------
A Federated Learning participant ("bank"). Run ONE process per bank. You can
run several on one desktop and/or on other desktops (see README).

Each round a client:
  1. Downloads the current global model from the server.
  2. Trains it locally on ONLY its own private shard of creditcard.csv using
     ADAPTIVE per-sample DP-SGD (common/dp_sgd.py):
        per-sample gradient clipping (EMA-tracked thresholds)
        + Gaussian noise whose layer-wise split follows Fisher-style importance
        + a Renyi-DP accountant / privacy filter that tracks epsilon.
  3. Uploads the resulting model weights (never raw data) and its epsilon.

Example:
    python client/client.py --server http://192.168.1.10:5000 --client_id 0 --num_clients 4
"""

import argparse
import os
import sys
import time

import numpy as np
import requests

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(PROJECT_ROOT)
from common.data_utils import (fit_scaler, load_full_dataset, make_global_test_split,
                               partition_client_shard, preprocess)
from common.dp_sgd import LocalDPTrainer
from common.model import FraudMLP
from common.serialization import decode_state, encode_state


def http(method, url, retries=60, wait=3.0, **kwargs):
    """requests wrapper that keeps retrying if the server is not reachable yet
    (e.g. you started the client before the server, or the network blips)."""
    for attempt in range(retries):
        try:
            r = requests.request(method, url, timeout=120, **kwargs)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"[client] cannot reach server ({type(e).__name__}); retry {attempt + 1}/{retries} in {wait}s ...")
            time.sleep(wait)
    raise SystemExit("[client] server unreachable. Check IP/port/firewall (see README).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="e.g. http://192.168.1.10:5000")
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--num_clients", type=int, required=True)
    parser.add_argument("--data_csv", default="creditcard.csv")
    parser.add_argument("--epochs", type=int, default=2, help="local epochs per round")
    parser.add_argument("--batch_size", type=int, default=256, help="expected (Poisson) batch size")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--pos_weight", type=float, default=5.0, help="up-weight the rare fraud class in the loss")
    parser.add_argument("--noise_multiplier", type=float, default=1.0,
                        help="overall DP noise multiplier z. HIGHER = more private (smaller epsilon), less accurate")
    parser.add_argument("--adaptive", type=int, default=1,
                        help="1 = adaptive DP (this project). 0 = fixed-clip/equal-noise baseline")
    parser.add_argument("--dp", type=int, default=1,
                        help="1 = differential privacy ON. 0 = NO privacy (reference run, epsilon undefined)")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--epsilon_budget", type=float, default=8.0,
                        help="privacy filter: stop local training before epsilon would exceed this (<=0 disables)")
    parser.add_argument("--seed", type=int, default=42, help="MUST match the server's seed")
    parser.add_argument("--run_seed", type=int, default=0,
                        help="changes ONLY the randomness of training/noise (not the data split); use different values for repeated runs")
    parser.add_argument("--non_iid", type=int, default=0)
    parser.add_argument("--poll_interval", type=float, default=2.0)
    args = parser.parse_args()

    csv_path = args.data_csv if os.path.isabs(args.data_csv) or os.path.exists(args.data_csv) \
        else os.path.join(PROJECT_ROOT, args.data_csv)
    df = load_full_dataset(csv_path)
    train_df, _ = make_global_test_split(df, test_size=0.2, seed=args.seed)
    scaler = fit_scaler(train_df)
    shard_df = partition_client_shard(train_df, args.client_id, args.num_clients,
                                      seed=args.seed, non_iid=bool(args.non_iid))
    X, y = preprocess(shard_df, scaler)
    mode = "NO-DP reference" if not args.dp else ("ADAPTIVE DP" if args.adaptive else "FIXED-BASELINE DP")
    print(f"[client {args.client_id}] local shard: {len(X)} samples, {int(y.sum())} fraud "
          f"({100 * y.mean():.3f}% positive) | mode={mode} | noise_multiplier={args.noise_multiplier}")

    model = FraudMLP(seed=args.seed)
    trainer = LocalDPTrainer(
        n_samples=len(X), batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        pos_weight=args.pos_weight, noise_multiplier=args.noise_multiplier,
        adaptive=bool(args.adaptive), dp=bool(args.dp), delta=args.delta,
        epsilon_budget=args.epsilon_budget if args.epsilon_budget > 0 else None,
        seed=args.seed * 1000 + args.client_id + 100000 * args.run_seed)

    http("POST", f"{args.server}/register", json={"client_id": args.client_id})

    while True:
        status = http("GET", f"{args.server}/status")
        if status["finished"]:
            print(f"[client {args.client_id}] server reports training finished. Exiting.")
            break

        resp = http("GET", f"{args.server}/get_global_model")
        server_round = resp["round"]
        model.set_state(decode_state(resp["model_state"]))

        print(f"[client {args.client_id}] -- round {server_round}: local DP training on {len(X)} samples --")
        info = trainer.train_round(model, X, y)

        eps = info["epsilon"]
        payload = {
            "client_id": args.client_id,
            "round": server_round,
            "model_state": encode_state(model.get_state()),
            "num_samples": len(X),
            "epsilon": None if eps != eps else eps,
            "exhausted": info["exhausted"],
        }
        submit_resp = http("POST", f"{args.server}/submit_update", json=payload)
        print(f"[client {args.client_id}] submitted round {server_round} "
              f"(my epsilon so far: {eps:.3f}): {submit_resp['status']}")

        while True:     # wait until this round is aggregated
            time.sleep(args.poll_interval)
            status = http("GET", f"{args.server}/status")
            if status["finished"] or status["round"] > server_round:
                break


if __name__ == "__main__":
    main()
