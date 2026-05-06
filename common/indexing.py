from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_index_maps(
        train_df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "movie_id",
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    user_ids = sorted(train_df[user_col].dropna().astype(int).unique())
    item_ids = sorted(train_df[item_col].dropna().astype(int).unique())

    user2idx = {u: idx for idx, u in enumerate(user_ids)}
    idx2user = {idx: u for u, idx in user2idx.items()}

    item2idx = {i: idx for idx, i in enumerate(item_ids)}
    idx2item = {idx: i for i, idx in item2idx.items()}

    return user2idx, idx2user, item2idx, idx2item


def build_interaction_matrix(
        train_df: pd.DataFrame,
        user2idx: Dict[int, int],
        item2idx: Dict[int, int],
        user_col: str = "user_id",
        item_col: str = "movie_id",
        value: float = 1.0,
) -> csr_matrix:
    df = train_df[[user_col, item_col]].copy()
    df[user_col] = df[user_col].map(user2idx)
    df[item_col] = df[item_col].map(item2idx)

    if df[user_col].isna().any() or df[item_col].isna().any():
        bad_rows = train_df.loc[df[user_col].isna() | df[item_col].isna(), [user_col, item_col]]
        raise ValueError(
            "Found ids absent in index maps while building interaction matrix.\n"
            f"Examples:\n{bad_rows.head()}"
        )

    rows = df[user_col].to_numpy(dtype=np.int64)
    cols = df[item_col].to_numpy(dtype=np.int64)
    data = np.full(len(df), value, dtype=np.float32)

    return csr_matrix(
        (data, (rows, cols)),
        shape=(len(user2idx), len(item2idx)),
        dtype=np.float32,
    )
