"""
scripts/run_experiments.py
----------------------------
Runs the report experiments for you, ALL ON ONE MACHINE (server + N client
processes started automatically), one after another, and draws the comparison
plots. Use it for (a) the dry run, and (b) generating your result graphs.
For the live two-desktop demo you start the server/clients by hand instead
(see README).

    python scripts/run_experiments.py                       # nodp, fixed, adaptive, 15 rounds, 4 clients
    python scripts/run_experiments.py --repeats 3           # 3 repetitions of each (recommended for the report)
    python scripts/run_experiments.py --experiments adaptive --rounds 5   # quick test

Results: results/<experiment>_run<k>/metrics.csv  and plots in results/plots/
"""
import argparse, os, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXPERIMENTS = {
    "nodp":     ["--dp", "0"],
    "fixed":    ["--dp", "1", "--adaptive", "0"],
    "adaptive": ["--dp", "1", "--adaptive", "1"],
}
LABELS = {"nodp": "No DP", "fixed": "Fixed DP", "adaptive": "Adaptive DP"}

p = argparse.ArgumentParser()
p.add_argument("--experiments", nargs="+", default=["nodp", "fixed", "adaptive"], choices=list(EXPERIMENTS))
p.add_argument("--clients", type=int, default=4)
p.add_argument("--rounds", type=int, default=15)
p.add_argument("--repeats", type=int, default=1)
p.add_argument("--noise_multiplier", type=float, default=1.0)
p.add_argument("--port", type=int, default=5055)
p.add_argument("--epsilon_budget", type=float, default=8.0, help="privacy filter budget (clients + server)")
p.add_argument("--extra", default="", help='extra client flags as ONE quoted string, e.g. --extra "--lr 0.005"')
a = p.parse_args()

py = sys.executable
done = []
for rep in range(1, a.repeats + 1):
    for exp in a.experiments:
        name = f"{exp}_run{rep}"
        rdir = os.path.join("results", name)
        print(f"\n===== {name}: starting server + {a.clients} clients =====", flush=True)
        server = subprocess.Popen([py, os.path.join(ROOT, "server", "server.py"), "--port", str(a.port),
                                   "--num_clients", str(a.clients), "--num_rounds", str(a.rounds),
                                   "--results_dir", rdir, "--epsilon_budget", str(a.epsilon_budget),
                                   "--model_seed", str(42 + rep - 1)], cwd=ROOT)
        time.sleep(15)       # give the server time to load the dataset
        clients = [subprocess.Popen([py, os.path.join(ROOT, "client", "client.py"),
                                     "--server", f"http://127.0.0.1:{a.port}", "--client_id", str(i),
                                     "--num_clients", str(a.clients),
                                     "--noise_multiplier", str(a.noise_multiplier),
                                     "--epsilon_budget", str(a.epsilon_budget),
                                     "--run_seed", str(rep - 1)]
                                    + EXPERIMENTS[exp] + a.extra.split(), cwd=ROOT,
                                    stdout=subprocess.DEVNULL) for i in range(a.clients)]
        for c in clients:
            c.wait()
        server.wait()
        done.append((exp, os.path.join(ROOT, rdir, "metrics.csv")))
        print(f"===== {name} finished -> {rdir}/metrics.csv =====", flush=True)

# plots: one line per experiment (last repeat), summary averages if repeats > 1
runs, labels = [], []
for exp in a.experiments:
    path = [m for e, m in done if e == exp][-1]
    runs.append(path); labels.append(LABELS[exp])
subprocess.run([py, os.path.join(ROOT, "evaluation", "plot_results.py"), "--runs", *runs,
                "--labels", *labels, "--out", os.path.join(ROOT, "results", "plots")], cwd=ROOT)
if a.repeats > 1:
    import pandas as pd
    rows = []
    for exp in a.experiments:
        finals = [pd.read_csv(m).iloc[-1] for e, m in done if e == exp]
        df = pd.DataFrame(finals)[["auc", "pr_auc", "recall", "precision", "f1", "epsilon"]]
        rows.append({"run": LABELS[exp], **{f"{c}_mean": df[c].mean() for c in df}, **{f"{c}_std": df[c].std() for c in ["auc", "pr_auc"]}})
    out = os.path.join(ROOT, "results", "plots", "summary_over_repeats.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False)); print("saved:", out)
