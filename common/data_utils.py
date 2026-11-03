"""
common/data_utils.py
----------------------
Loading, preprocessing, and client-partitioning for the Kaggle
"Credit Card Fraud Detection" dataset (creditcard.csv: Time, V1..V28,
Amount, Class).

IMPORTANT DESIGN CHOICE FOR THE TWO-DESKTOP SETUP:
Every desktop (server AND every client) must have an identical copy of
creditcard.csv. Each process independently re-derives:
  1) the same global train/test split (fixed seed -> identical split
     everywhere, no need to transmit the test set),
  2) the same StandardScaler fit on that global train split (fixed seed
     -> identical scaling everywhere, no need to transmit the scaler),
  3) a *different, non-overlapping* shard of the train split for each
     client, selected purely by client_id/num_clients (so client 0 and
     client 1 never train on the same rows, just like separate banks
     each holding their own private transactions).
This means nothing about the raw data itself ever has to be transmitted
over the network -- only model weights/updates are -- which is the whole
point of Federated Learning.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABEL_COLUMN = "Class"


def load_full_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def make_global_test_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[LABEL_COLUMN]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[["Time", "Amount"]])
    return scaler


def preprocess(df: pd.DataFrame, scaler: StandardScaler):
    X = df[FEATURE_COLUMNS].copy()
    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])
    y = df[LABEL_COLUMN].astype(int).values
    return X.values.astype(np.float32), y


def partition_client_shard(train_df: pd.DataFrame, client_id: int, num_clients: int,
                            seed: int = 42, non_iid: bool = False) -> pd.DataFrame:
    """Deterministically returns THIS client's non-overlapping shard of the
    training split. Every desktop computes this independently and arrives
    at the same answer, as long as they run with the same seed/num_clients
    and have the same creditcard.csv.

    non_iid=True produces a skewed, more realistic "different institutions
    see different transaction profiles" split by sorting on Amount before
    slicing, instead of a uniform random shuffle.
    """
    if client_id < 0 or client_id >= num_clients:
        raise ValueError(f"client_id must be in [0, {num_clients - 1}]")

    if non_iid:
        ordered = train_df.sort_values("Amount", kind="stable").reset_index(drop=True)
    else:
        ordered = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    index_chunks = np.array_split(np.arange(len(ordered)), num_clients)
    return ordered.iloc[index_chunks[client_id]].reset_index(drop=True)
