"""
data/inspect_shard.py
-----------------------
Optional sanity check. Prints (and optionally exports) the private data shard
a given client_id would train on, so you can verify -- and show your examiner --
that different clients hold different, non-overlapping data.

    python data/inspect_shard.py --num_clients 4
    python data/inspect_shard.py --num_clients 4 --non_iid 1
"""
import argparse, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.data_utils import load_full_dataset, make_global_test_split, partition_client_shard

p = argparse.ArgumentParser()
p.add_argument("--data_csv", default="creditcard.csv")
p.add_argument("--num_clients", type=int, required=True)
p.add_argument("--non_iid", type=int, default=0)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

df = load_full_dataset(a.data_csv)
train_df, test_df = make_global_test_split(df, seed=a.seed)
print(f"Full dataset: {len(df)} rows | global train: {len(train_df)} | global test (server only): {len(test_df)}\n")
print(f"{'client':<8}{'rows':>9}{'fraud':>8}{'fraud %':>10}{'mean Amount':>14}")
for cid in range(a.num_clients):
    s = partition_client_shard(train_df, cid, a.num_clients, seed=a.seed, non_iid=bool(a.non_iid))
    print(f"{cid:<8}{len(s):>9}{int(s['Class'].sum()):>8}{100*s['Class'].mean():>9.3f}%{s['Amount'].mean():>14.2f}")
