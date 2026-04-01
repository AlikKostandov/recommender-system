from __future__ import annotations

from typing import Tuple
import pandas as pd
import numpy as np


def time_split_last_n(
    data: pd.DataFrame,
    last_n: int = 1,
    user_col: str = "user_id",
    time_col: str = "datetime",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-last-n-out split per user based on timestamp."""
    if time_col not in data.columns:
        raise ValueError(f"Column '{time_col}' is required for time split")

    df = data.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values([user_col, time_col])

    test = df.groupby(user_col, group_keys=False).tail(last_n).copy()
    train = df.drop(index=test.index).copy()

    # Keep only users that still have at least one train interaction.
    good_users = train[user_col].unique()
    test = test[test[user_col].isin(good_users)].copy()

    return train.reset_index(drop=True), test.reset_index(drop=True)


def temporal_train_test_split(
    df_pos: pd.DataFrame,
    user_col: str = "user_id",
    time_col: str = "datetime",
    test_ratio: float = 0.2,
    min_train: int = 1,
    min_test: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df_pos.sort_values([user_col, time_col]).copy()

    train_parts = []
    test_parts = []

    for _, grp in df_sorted.groupby(user_col, sort=False):
        n = len(grp)
        if n < (min_train + min_test):
            continue

        test_size = max(min_test, int(np.ceil(n * test_ratio)))
        train_size = n - test_size

        if train_size < min_train:
            train_size = min_train
            test_size = n - train_size

        if test_size < min_test:
            continue

        train_parts.append(grp.iloc[:train_size])
        test_parts.append(grp.iloc[train_size:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, test_df