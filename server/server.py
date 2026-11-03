"""
server/server.py
------------------
The Federated Learning coordinator. Run this ONCE, on ONE desktop
("Desktop A" / the server machine).

Responsibilities:
  - Holds the single global model.
  - Serves the current global model to any client that asks (GET /get_global_model).
  - Accepts each client's locally-trained, clipped, noised model update for
    the current round (POST /submit_update).
  - Once every expected client has submitted for a round, performs FedAvg
    aggregation (weighted by each client's local sample count), updates the
    global model, evaluates it on a held-out global test set, logs metrics,
    advances the privacy accountant, and opens the next round.
  - Exposes GET /status for clients (and you) to poll training progress.

Start it with:
    python server/server.py --config server/config.yaml
(run from the project root so the relative paths in config.yaml resolve.)
"""

import argparse
import csv
import os
import sys
import threading

import numpy as np
import yaml
from flask import Flask, jsonify, request
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(PROJECT_ROOT)
from common.data_utils import fit_scaler, load_full_dataset, make_global_test_split, preprocess
from common.model import FraudMLP, LAYER_NAMES
from common.serialization import decode_state, encode_state

app = Flask(__name__)

STATE = {
    "lock": threading.Lock(),
    "round": 0,
    "num_clients": None,
    "received_updates": {},
    "model": None,
    "test_X": None,
    "test_y": None,
    "epsilon": float("nan"),
    "epsilon_budget": None,
    "auto_shutdown": True,
    "results_path": None,
    "max_rounds": None,
    "finished": False,
}


def evaluate_global_model():
    probs = STATE["model"].predict_proba(STATE["test_X"])
    preds = (probs >= 0.5).astype(int)
    y_true = STATE["test_y"]
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "auc": roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": average_precision_score(y_true, probs),
    }


def fedavg_aggregate(updates):
    """updates: list of (state_dict_of_numpy_arrays, num_samples)"""
    total = sum(n for _, n in updates)
    agg = None
    for state_np, n in updates:
        weight = n / total
        if agg is None:
            agg = {k: state_np[k] * weight for k in LAYER_NAMES}
        else:
            for k in LAYER_NAMES:
                agg[k] = agg[k] + state_np[k] * weight
    return agg


@app.route("/register", methods=["POST"])
def register():
    data = request.json
    print(f"[server] client {data.get('client_id')} registered.")
    return jsonify({"status": "ok"})


@app.route("/get_global_model", methods=["GET"])
def get_global_model():
    with STATE["lock"]:
        state_np = STATE["model"].get_state()
        return jsonify({
            "round": STATE["round"],
            "finished": STATE["finished"],
            "model_state": encode_state(state_np),
        })


@app.route("/submit_update", methods=["POST"])
def submit_update():
    payload = request.json
    client_id = payload["client_id"]
    client_round = payload["round"]

    with STATE["lock"]:
        if STATE["finished"]:
            return jsonify({"status": "finished"})
        if client_round != STATE["round"]:
            return jsonify({"status": "stale_round", "current_round": STATE["round"]})
        if client_id in STATE["received_updates"]:
            return jsonify({"status": "already_submitted"})

        STATE["received_updates"][client_id] = {
            "state": decode_state(payload["model_state"]),
            "num_samples": payload["num_samples"],
            "epsilon": payload.get("epsilon"),
            "exhausted": bool(payload.get("exhausted", False)),
        }
        print(f"[server] received update from client {client_id} for round {STATE['round']} "
              f"({len(STATE['received_updates'])}/{STATE['num_clients']})")

        if len(STATE["received_updates"]) >= STATE["num_clients"]:
            _aggregate_round()

        return jsonify({"status": "received"})


def _aggregate_round():
    updates = [(v["state"], v["num_samples"]) for v in STATE["received_updates"].values()]
    new_state = fedavg_aggregate(updates)
    STATE["model"].set_state(new_state)

    # Privacy: every client runs DP-SGD on ITS OWN records and tracks its own epsilon
    # (RDP accountant). Shards are disjoint, so by parallel composition the overall
    # record-level epsilon is the WORST (largest) client epsilon.
    all_clients = list(STATE["received_updates"].values())
    eps_list = [c["epsilon"] for c in all_clients if c["epsilon"] is not None and c["epsilon"] == c["epsilon"]]
    eps = max(eps_list) if eps_list else float("nan")
    STATE["epsilon"] = eps
    any_exhausted = any(c["exhausted"] for c in all_clients)

    metrics = evaluate_global_model()
    metrics.update({"round": STATE["round"], "epsilon": eps,
                     "num_clients_this_round": len(STATE["received_updates"])})
    _log_metrics(metrics)
    print(f"[server] === round {STATE['round']} complete | auc={metrics['auc']:.4f} "
          f"pr_auc={metrics['pr_auc']:.4f} recall={metrics['recall']:.4f} precision={metrics['precision']:.4f} "
          f"f1={metrics['f1']:.4f} acc={metrics['accuracy']:.4f} | epsilon so far={eps:.3f} ===")

    STATE["received_updates"] = {}
    STATE["round"] += 1
    budget_hit = any_exhausted or (STATE["epsilon_budget"] is not None and eps == eps
                                   and eps >= STATE["epsilon_budget"])
    if STATE["round"] >= STATE["max_rounds"] or budget_hit:
        STATE["finished"] = True
        out_dir = os.path.dirname(STATE["results_path"])
        np.savez(os.path.join(out_dir, "global_model_final.npz"), **STATE["model"].get_state())
        why = "PRIVACY BUDGET REACHED (privacy filter)" if budget_hit else "all rounds completed"
        print(f"[server] TRAINING FINISHED: {why}.")
        print(f"[server] metrics saved to : {STATE['results_path']}")
        print(f"[server] final model saved: {os.path.join(out_dir, 'global_model_final.npz')}")
        if STATE["auto_shutdown"]:
            print("[server] shutting down in 10 seconds (clients are being told training is finished) ...")
            threading.Timer(10.0, lambda: os._exit(0)).start()


def _log_metrics(metrics):
    file_exists = os.path.exists(STATE["results_path"])
    with open(STATE["results_path"], "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


@app.route("/status", methods=["GET"])
def status():
    with STATE["lock"]:
        return jsonify({
            "round": STATE["round"],
            "max_rounds": STATE["max_rounds"],
            "finished": STATE["finished"],
            "clients_submitted_this_round": len(STATE["received_updates"]),
            "num_clients_expected": STATE["num_clients"],
            "current_epsilon": None if STATE["epsilon"] != STATE["epsilon"] else STATE["epsilon"],
        })


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "server", "config.yaml"))
    parser.add_argument("--port", type=int)
    parser.add_argument("--num_clients", type=int)
    parser.add_argument("--num_rounds", type=int)
    parser.add_argument("--results_dir")
    parser.add_argument("--epsilon_budget", type=float)
    parser.add_argument("--model_seed", type=int, help="seed of the random starting model (default: config seed)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for k in ("port", "num_clients", "num_rounds", "results_dir", "epsilon_budget"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    df = load_full_dataset(_abs(cfg["data_csv"]))
    train_df, test_df = make_global_test_split(df, test_size=cfg["test_size"], seed=cfg["seed"])
    scaler = fit_scaler(train_df)
    X_test, y_test = preprocess(test_df, scaler)

    STATE["model"] = FraudMLP(seed=args.model_seed if args.model_seed is not None else cfg["seed"])
    STATE["num_clients"] = cfg["num_clients"]
    STATE["max_rounds"] = cfg["num_rounds"]
    STATE["test_X"], STATE["test_y"] = X_test, y_test
    STATE["epsilon_budget"] = cfg.get("epsilon_budget")
    STATE["auto_shutdown"] = cfg.get("auto_shutdown", True)

    results_dir = _abs(cfg["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    STATE["results_path"] = os.path.join(results_dir, "metrics.csv")
    if os.path.exists(STATE["results_path"]):      # each run starts fresh
        os.remove(STATE["results_path"])

    print(f"[server] global test set: {len(X_test)} samples ({int(y_test.sum())} fraud)")
    print(f"[server] results will be written to: {results_dir}")
    print(f"[server] starting on {cfg['host']}:{cfg['port']} | expecting {cfg['num_clients']} "
          f"client(s) per round for up to {cfg['num_rounds']} rounds | epsilon budget: {STATE['epsilon_budget']}")
    app.run(host=cfg["host"], port=cfg["port"], threaded=True)


if __name__ == "__main__":
    main()
