# Adaptive Differential Privacy for Federated Credit Card Fraud Detection
### Client–Server implementation (runs across two physical desktops)

Several "banks" (clients) collaboratively train a fraud detector on the Kaggle `creditcard.csv` data **without sharing
raw transactions**. Each bank trains with **adaptive per-sample DP-SGD** (clipping + calibrated noise) and a central
server averages the models (FedAvg), grades them on a held-out test set, and tracks the privacy budget (ε).

> **Version note:** this version replaces an earlier design that noised whole weight-updates (it did not learn: AUC
> jumped between 0.36 and 0.81 and ended at 0.47). Privacy is now applied *during local training, per sample*, as the
> methodology describes ("dynamic gradient clipping during local training ... EMA ... layer-wise Fisher ... privacy filter").

---

## 1. Project layout

```
fl_project/
├── README.md
├── requirements.txt
├── creditcard.csv                 <- YOU copy this in (identical file on BOTH desktops)
├── common/
│   ├── model.py                   NumPy MLP 30-24-12-1 (1,057 params), Adam
│   ├── dp_sgd.py                  LOCAL TRAINING with adaptive per-sample DP-SGD  (core)
│   ├── adaptive_dp.py             AdaptiveClipper (EMA), FisherImportance, RDP accountant / privacy filter (core)
│   ├── data_utils.py              split, scaling, per-client shards
│   └── serialization.py           weights <-> text for HTTP
├── server/  server.py, config.yaml
├── client/  client.py
├── scripts/run_experiments.py     runs nodp / fixed / adaptive experiments on ONE machine and makes the plots
├── evaluation/plot_results.py     graphs + summary table from metrics.csv files
├── data/inspect_shard.py          shows each client's private data shard
└── results/                       created automatically
```

## 2. How it works

```
   DEVICE A (server)                                    DEVICE B (clients)
 ┌────────────────────────────┐   1. get global model   ┌─────────────────────────────────────┐
 │ server.py :5000            │ ───────────────────────►│ client.py  (client 2)               │
 │  FedAvg, test-set grading, │                         │ client.py  (client 3)               │
 │  epsilon tracking,         │◄─────────────────────── │   2. DP-SGD on OWN private shard    │
 │  metrics.csv               │   3. upload weights+eps │      (clip, noise, track epsilon)   │
 │ (+ optional clients 0,1)   │                         └─────────────────────────────────────┘
 └────────────────────────────┘
```

**Adaptive DP (maps to your methodology)**

| Methodology | Code | What it does |
|---|---|---|
| Dynamic clipping in local training, EMA of historical gradient patterns | `AdaptiveClipper` | Per-layer clip threshold follows the (privately estimated) median per-sample gradient norm, smoothed by an EMA |
| Layer-wise Fisher information → less noise on important parameters | `FisherImportance` + `LocalDPTrainer._round_sigmas` | Fixed per-step privacy budget is split across the 6 layer tensors ∝ importance; important layers get less noise |
| Component-wise sensitivity | per-layer clip C_j, noise σ_j = z_j·C_j | Noise calibrated to each layer's own sensitivity |
| Privacy filter, realised loss in real time, stop training | `RDPAccountant` + `epsilon_budget` | Rényi-DP accountant; training stops before ε would exceed the budget |
| FedAvg | `server.py` | sample-size-weighted average |

Adaptive and fixed spend the **same nominal budget** (same noise multiplier `z`), so comparisons are fair; the accountant reports the true ε of each.

---

## 3. What must be on EACH desktop

| Item | Server desktop | Client desktop |
|---|---|---|
| Python 3.9–3.12 | ✅ | ✅ |
| Whole `fl_project/` folder | ✅ | ✅ |
| `creditcard.csv` in the project folder (**byte-identical**) | ✅ | ✅ |
| `pip install -r requirements.txt` (no GPU / PyTorch needed) | ✅ | ✅ |
| Same Wi-Fi/LAN | ✅ | ✅ |
| Inbound TCP port 5000 allowed in firewall | ✅ required | – |
| Runs | `server.py` (+ optional clients) | one or more `client.py` |

## 4. Step by step

### Phase 0 – setup (both desktops)
```bash
cd fl_project
python -m venv venv
venv\Scripts\activate            # Windows      (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```
Copy `creditcard.csv` into `fl_project/`. Verify identical files:
`certutil -hashfile creditcard.csv SHA256` (Windows) / `shasum -a 256 creditcard.csv` (Linux/Mac).
Optional: `python data/inspect_shard.py --num_clients 4` (shows the four clients hold different data).

### Phase 1 – dry run on ONE machine (do this first)
Quick test, ~3 minutes:
```bash
python scripts/run_experiments.py --experiments adaptive nodp --rounds 3
```
It starts a server and 4 clients by itself, runs both experiments, and prints where results were saved.
Healthy signs: lines like `=== round 2 complete | auc=… ===`, and at the end `TRAINING FINISHED … metrics saved to: <full path>`.

### Phase 2 – real two-desktop run
1. **Server desktop – find its IP:** `ipconfig` (Windows, "IPv4 Address") / `hostname -I` (Linux) / `ipconfig getifaddr en0` (Mac). Example: `192.168.1.7`.
2. **Server desktop – open port 5000.** Windows: open **PowerShell as Administrator** (right-click → *Run as administrator*) and run
   `New-NetFirewallRule -DisplayName "FL Server" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow`.
   (If you see "Access is denied" you were not admin. Alternatives: click **Allow** on Windows' first-run popup for Python, or add the rule in
   *Windows Defender Firewall with Advanced Security → Inbound Rules → New Rule → Port → TCP 5000*.) Ubuntu: `sudo ufw allow 5000/tcp`.
   Also make sure the Wi-Fi is set to a **Private** network.
3. **Server desktop – start the server** (leave it running):
   ```bash
   python server/server.py
   ```
   It prints `results will be written to: <full path>`.
4. **Client desktop – test connectivity:** open `http://192.168.1.7:5000/status` in a browser → you should see JSON.
5. **Client desktop – start clients** (one terminal each, venv activated; use the server's IP):
   ```bash
   python client/client.py --server http://192.168.1.7:5000 --client_id 2 --num_clients 4
   python client/client.py --server http://192.168.1.7:5000 --client_id 3 --num_clients 4
   ```
6. **Server desktop – two more terminals** for clients 0 and 1:
   ```bash
   python client/client.py --server http://127.0.0.1:5000 --client_id 0 --num_clients 4
   python client/client.py --server http://127.0.0.1:5000 --client_id 1 --num_clients 4
   ```
   (Or run all four on the client desktop with ids 0–3.) `--num_clients` must equal `num_clients` in `server/config.yaml` (default 4)
   and every `client_id` must be unique.
7. **Watch.** The server prints one line per round with AUC, PR-AUC, recall, precision, F1 and the **epsilon** spent. Clients print per-epoch loss and their own ε.
8. **It stops by itself.** After the last round (or when the privacy budget is reached) **the clients exit and the server shuts itself down ~10 s later – this is normal, not a crash.** The server prints the full paths of the saved files.

**Where are my results?** In the `results/` folder **inside the project folder** (always, regardless of where you launched from; the server prints the full path):
`results/metrics.csv` (one row per round) and `results/global_model_final.npz` (the final 1,057 weights).

### Phase 3 – experiments for the report
Each run starts a fresh `metrics.csv`, so either use the automatic script (recommended, one machine):
```bash
python scripts/run_experiments.py --repeats 3
```
(= No-DP, Fixed-DP, Adaptive-DP × 3 repetitions, 15 rounds, 4 clients, noise multiplier 1.0; writes `results/<exp>_run<k>/metrics.csv`,
plots in `results/plots/`, and `summary_over_repeats.csv` with mean/std. Each repeat uses a different random starting model and different training noise (`--model_seed` / `--run_seed`) but the same data split; the line graphs show the last repeat, the averages are in the CSV. Roughly 30+ minutes.) Other settings: `--noise_multiplier 2.0`, `--rounds 20`, `--clients 6`.

…or do manual runs on the two desktops with these client flags (same flags on **all** clients), moving `results/metrics.csv` aside after each run:

| Run | Client flags |
|---|---|
| Adaptive DP (**proposed**) | *(defaults)* |
| Fixed DP baseline | `--adaptive 0` |
| No privacy (reference) | `--dp 0` |
| Stronger privacy | `--noise_multiplier 2.0` |
| Show the privacy filter stopping training | `--epsilon_budget 0.9` |
| Non-IID banks | `--non_iid 1` |

Then:
```bash
python evaluation/plot_results.py --runs results_nodp/metrics.csv results_fixed/metrics.csv results_adaptive/metrics.csv --labels "No DP" "Fixed DP" "Adaptive DP"
```

---

## 5. What results to expect (measured by me — your numbers will differ slightly)

Final global model after 15 rounds, 4 clients, whole test set (56,962 transactions, 98 fraud). Mean over seeds, simulated with exactly this code:

| Setting | AUC | PR-AUC | ε (δ=1e-5) |
|---|---|---|---|
| No privacy (1 run) | 0.984 | 0.838 | – |
| Fixed DP, z=1 (4 seeds) | 0.875 | 0.58 | 1.62 |
| **Adaptive DP, z=1 (4 seeds)** | **0.901** | **0.64** | 1.66 |
| Fixed DP, z=2 (3 seeds) | 0.849 | 0.58 | 0.73 |
| **Adaptive DP, z=2 (3 seeds)** | **0.878** | 0.60 | 0.79 |

How to read this honestly:
* Privacy costs accuracy: ~0.98 → ~0.90 AUC at ε≈1.7. That is a real, reportable trade-off.
* Adaptive had the higher AUC in 3 of 4 seeds (z=1) and 3 of 3 (z=2), but margins are a few AUC points, comparable to seed-to-seed variation. Describe it as "promising/consistent in my runs", not "proven better". Run your own repeats.
* Adaptive spends slightly **more** ε (+3–8%) because tracking the clip threshold privately costs a little budget.
* **The first 2–3 rounds look poor** (AUC can even be < 0.5) with DP on; it stabilises by round ~4. I observed this but did not fully diagnose it (clipping bias against the rare, large-gradient fraud samples is my hypothesis). Show the whole curve and explain it.
* Accuracy is meaningless here (99.8% are legitimate). Quote AUC, PR-AUC, recall, precision.
* My 15-round numbers come from an in-process simulation of this same code; the HTTP pipeline was tested end-to-end for shorter runs. **Quote only the numbers from your own runs.**

## 6. Limitations to state yourself
1. **Guarantee scope:** record-level (ε, δ)-DP for each bank's own data via per-sample DP-SGD, composed over rounds with a Rényi accountant. Banks hold disjoint shards, so the overall ε is the **worst client's ε** (parallel composition). The server trusts clients' reported ε (no verification).
2. Clip thresholds and layer importances are derived from noised signals only (post-processing), so they are privacy-safe; the noisy clip-tracking counts are charged to ε.
3. Laplace noise is **not** implemented in this version (Gaussian/RDP only).
4. **No secure aggregation**, no HTTPS (plain HTTP on a trusted LAN; weights use `pickle` → never expose publicly), no poisoning defence.
5. Synchronous rounds: a crashed client stalls the server.
6. Default hyper-parameters were chosen with quick checks that used the test split; for stricter work, tune on a validation split.
7. Banks are simulated by splitting one public dataset; PCA-anonymised 2013 data.

## 7. Troubleshooting
| Symptom | Fix |
|---|---|
| `New-NetFirewallRule: Access is denied` | PowerShell not run as Administrator – see Phase 2 step 2 |
| Client: `cannot reach server … retry` | wrong IP/port, server not started, different networks, firewall |
| `Address already in use` | an old server still running; close it or change `port` |
| `FileNotFoundError: creditcard.csv` | put the CSV in the project folder (or pass `--data_csv`, or edit `data_csv` in config) |
| Server stuck at `(n/4)` | fewer client processes than `num_clients` – start the missing ones |
| "Nothing in results" | look in `<project folder>/results/` (full path printed by the server); the server also deletes old `metrics.csv` at start |
| Clients exited / server closed | normal after the last round |
| Training ends early with "PRIVACY BUDGET REACHED" | the privacy filter did its job; raise `epsilon_budget` (server config + `--epsilon_budget` on clients) |
| Slow | lower `--epochs`; close other programs |
