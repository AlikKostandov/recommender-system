from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Set, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class ColdStartSplitResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    cold_users: Set[int]
    eligible_cold_users: Set[int]
    user_interactions_df: pd.DataFrame
    split_stats: pd.DataFrame
    config: Dict[str, Any]


def build_user_interaction_counts(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
) -> pd.DataFrame:
    """
    Считает число взаимодействий на пользователя.

    Returns
    -------
    pd.DataFrame
        Колонки:
        - user_id
        - n_interactions
    """
    if user_col not in interactions.columns:
        raise ValueError(f"Column '{user_col}' not found in interactions DataFrame.")

    counts_df = (
        interactions.groupby(user_col)
        .size()
        .rename("n_interactions")
        .reset_index()
        .sort_values("n_interactions", ascending=True)
        .reset_index(drop=True)
    )
    return counts_df


def summarize_interaction_buckets(
    user_counts_df: pd.DataFrame,
    count_col: str = "n_interactions",
) -> pd.DataFrame:
    """
    Формирует удобную сводку по диапазонам числа взаимодействий.
    """
    if count_col not in user_counts_df.columns:
        raise ValueError(f"Column '{count_col}' not found in user_counts_df.")

    counts = user_counts_df[count_col]
    total_users = len(user_counts_df)

    summary = pd.DataFrame({
        "segment": ["1", "2-3", "4-5", "6-10", "11-20", "21+"],
        "users_count": [
            (counts == 1).sum(),
            counts.between(2, 3).sum(),
            counts.between(4, 5).sum(),
            counts.between(6, 10).sum(),
            counts.between(11, 20).sum(),
            (counts >= 21).sum(),
        ]
    })
    summary["share_percent"] = (
        summary["users_count"] / total_users * 100
    ).round(2)

    return summary


def select_eligible_cold_users(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    min_interactions_for_cold: int = 15,
) -> Set[int]:
    """
    Выбирает пользователей, которых можно использовать для synthetic cold start.

    Логика:
    берем только пользователей, у которых достаточно истории,
    чтобы после искусственного урезания train у них остался meaningful test.
    """
    if min_interactions_for_cold < 2:
        raise ValueError("min_interactions_for_cold should be >= 2.")

    user_counts_df = build_user_interaction_counts(interactions, user_col=user_col)
    eligible_users = set(
        user_counts_df.loc[
            user_counts_df["n_interactions"] >= min_interactions_for_cold,
            user_col
        ].tolist()
    )
    return eligible_users


def sample_cold_users(
    eligible_cold_users: Iterable[int],
    cold_user_fraction: float = 0.2,
    random_state: int = 42,
) -> Set[int]:
    """
    Случайно выбирает подмножество пользователей, для которых будет смоделирован cold start.
    """
    if not (0.0 <= cold_user_fraction <= 1.0):
        raise ValueError("cold_user_fraction must be in [0, 1].")

    eligible_cold_users = list(eligible_cold_users)
    if len(eligible_cold_users) == 0:
        return set()

    rng = np.random.default_rng(random_state)
    n_cold_users = int(round(len(eligible_cold_users) * cold_user_fraction))

    if n_cold_users == 0 and cold_user_fraction > 0:
        n_cold_users = 1

    if n_cold_users > len(eligible_cold_users):
        n_cold_users = len(eligible_cold_users)

    sampled = rng.choice(eligible_cold_users, size=n_cold_users, replace=False)
    return set(sampled.tolist())


def temporal_user_split(
    user_df: pd.DataFrame,
    time_col: str,
    train_keep_n: Optional[int] = None,
    test_last_n: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Делит interactions одного пользователя по времени.

    Варианты:
    - train_keep_n: оставить в train первые N взаимодействий, остальные в test
    - test_last_n: оставить в test последние N взаимодействий, остальные в train

    Использовать только один из параметров.
    """
    if (train_keep_n is None and test_last_n is None) or (
        train_keep_n is not None and test_last_n is not None
    ):
        raise ValueError("Specify exactly one of: train_keep_n or test_last_n.")

    user_df = user_df.sort_values(time_col).copy()

    if train_keep_n is not None:
        if train_keep_n < 1:
            raise ValueError("train_keep_n must be >= 1.")
        train_part = user_df.iloc[:train_keep_n].copy()
        test_part = user_df.iloc[train_keep_n:].copy()
        return train_part, test_part

    if test_last_n is not None:
        if test_last_n < 1:
            raise ValueError("test_last_n must be >= 1.")

        if len(user_df) <= test_last_n:
            return user_df.copy(), user_df.iloc[0:0].copy()

        train_part = user_df.iloc[:-test_last_n].copy()
        test_part = user_df.iloc[-test_last_n:].copy()
        return train_part, test_part

    raise RuntimeError("Unexpected split configuration.")


def make_synthetic_cold_start_split(
    interactions: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "movie_id",
    time_col: str = "datetime",
    cold_user_fraction: float = 0.2,
    cold_n: int = 3,
    min_interactions_for_cold: int = 15,
    warm_last_n: int = 1,
    random_state: int = 42,
) -> ColdStartSplitResult:
    """
    Формирует общий synthetic cold start split.

    Для cold users:
    - в train остаются первые cold_n взаимодействий
    - остальные идут в test

    Для остальных пользователей:
    - обычный temporal split по warm_last_n
    """
    required_cols = {user_col, item_col, time_col}
    missing_cols = required_cols - set(interactions.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if cold_n < 1:
        raise ValueError("cold_n must be >= 1.")
    if warm_last_n < 1:
        raise ValueError("warm_last_n must be >= 1.")

    df = interactions.copy()
    df = df.sort_values([user_col, time_col]).reset_index(drop=True)

    user_interactions_df = build_user_interaction_counts(df, user_col=user_col)

    eligible_cold_users = select_eligible_cold_users(
        df,
        user_col=user_col,
        min_interactions_for_cold=min_interactions_for_cold,
    )

    cold_users = sample_cold_users(
        eligible_cold_users=eligible_cold_users,
        cold_user_fraction=cold_user_fraction,
        random_state=random_state,
    )

    train_parts = []
    test_parts = []
    split_rows = []

    for user_id, user_df in df.groupby(user_col, sort=False):
        original_count = len(user_df)

        if user_id in cold_users:
            train_user, test_user = temporal_user_split(
                user_df=user_df,
                time_col=time_col,
                train_keep_n=cold_n,
            )

            if len(test_user) == 0:
                # safety net
                train_user = user_df.iloc[:-1].copy()
                test_user = user_df.iloc[-1:].copy()

            split_type = "cold"
        else:
            train_user, test_user = temporal_user_split(
                user_df=user_df,
                time_col=time_col,
                test_last_n=warm_last_n,
            )
            split_type = "warm"

        train_parts.append(train_user)
        if len(test_user) > 0:
            test_parts.append(test_user)

        split_rows.append({
            user_col: user_id,
            "split_type": split_type,
            "original_interactions": original_count,
            "train_interactions": len(train_user),
            "test_interactions": len(test_user),
        })

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    split_stats = pd.DataFrame(split_rows)

    config = {
        "user_col": user_col,
        "item_col": item_col,
        "time_col": time_col,
        "cold_user_fraction": cold_user_fraction,
        "cold_n": cold_n,
        "min_interactions_for_cold": min_interactions_for_cold,
        "warm_last_n": warm_last_n,
        "random_state": random_state,
    }

    return ColdStartSplitResult(
        train_df=train_df,
        test_df=test_df,
        cold_users=cold_users,
        eligible_cold_users=eligible_cold_users,
        user_interactions_df=user_interactions_df,
        split_stats=split_stats,
        config=config,
    )


def get_eval_users(
    split_result: ColdStartSplitResult,
    mode: str = "cold_only",
    user_col: str = "user_id",
) -> Set[int]:
    """
    Возвращает множество пользователей, по которым считать метрики.

    mode:
    - cold_only
    - warm_only
    - all
    """
    if mode not in {"cold_only", "warm_only", "all"}:
        raise ValueError("mode must be one of: {'cold_only', 'warm_only', 'all'}")

    split_stats = split_result.split_stats

    if mode == "cold_only":
        return set(split_result.cold_users)

    if mode == "warm_only":
        return set(
            split_stats.loc[split_stats["split_type"] == "warm", user_col].tolist()
        )

    return set(split_stats[user_col].tolist())


def filter_interactions_by_users(
    df: pd.DataFrame,
    users: Iterable[int],
    user_col: str = "user_id",
) -> pd.DataFrame:
    """
    Оставляет только interactions заданных пользователей.
    """
    users = set(users)
    return df[df[user_col].isin(users)].copy()


def build_seen_items_map(
    train_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "movie_id",
) -> Dict[int, Set[int]]:
    return (
        train_df.groupby(user_col)[item_col]
        .apply(lambda x: set(x.tolist()))
        .to_dict()
    )


def build_ground_truth_map(
    test_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "movie_id",
) -> Dict[int, Set[int]]:
    """
    user_id -> множество релевантных test-айтемов.
    """
    return (
        test_df.groupby(user_col)[item_col]
        .apply(lambda x: set(x.tolist()))
        .to_dict()
    )


def summarize_split_result(
    split_result: ColdStartSplitResult,
    user_col: str = "user_id",
) -> pd.DataFrame:
    """
    Удобная компактная сводка по synthetic cold-start split.
    """
    split_stats = split_result.split_stats

    rows = []

    for split_type in ["cold", "warm"]:
        part = split_stats.loc[split_stats["split_type"] == split_type]

        rows.append({
            "group": f"{split_type}_users",
            "n_users": len(part),
            "mean_original_interactions": round(
                part["original_interactions"].mean(), 2
            ) if len(part) > 0 else 0.0,
            "mean_train_interactions": round(
                part["train_interactions"].mean(), 2
            ) if len(part) > 0 else 0.0,
            "mean_test_interactions": round(
                part["test_interactions"].mean(), 2
            ) if len(part) > 0 else 0.0,
        })

    rows.append({
        "group": "all_users",
        "n_users": split_stats[user_col].nunique(),
        "mean_original_interactions": round(
            split_stats["original_interactions"].mean(), 2
        ),
        "mean_train_interactions": round(
            split_stats["train_interactions"].mean(), 2
        ),
        "mean_test_interactions": round(
            split_stats["test_interactions"].mean(), 2
        ),
    })

    return pd.DataFrame(rows)