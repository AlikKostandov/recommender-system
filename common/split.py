from __future__ import annotations

from typing import Tuple
import pandas as pd
import numpy as np


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