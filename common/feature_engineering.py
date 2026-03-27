from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import pandas as pd


def _clean_token_value(value) -> Optional[str]:
    if pd.isna(value):
        return None

    value = str(value).strip()
    if not value:
        return None

    value = value.replace(" ", "_")
    return value


def _add_token(store: dict[int, set[str]], entity_id: int, prefix: str, value) -> None:
    clean_value = _clean_token_value(value)
    if clean_value is None:
        return

    store[int(entity_id)].add(f"{prefix}:{clean_value}")


def build_user_feature_tokens(
    user_features_base: pd.DataFrame,
    user_col: str = "user_id",
    include_gender: bool = True,
    include_occupation: bool = True,
    include_age_group: bool = True,
) -> Dict[int, List[str]]:
    """
    Build user feature tokens for LightFM.

    Example:
        {
            1: ["gender:M", "occupation:3", "age_group:2"],
            2: ["gender:F", "occupation:7", "age_group:4"],
        }
    """
    required_cols = {user_col}
    missing = required_cols - set(user_features_base.columns)
    if missing:
        raise ValueError(f"user_features_base is missing columns: {missing}")

    result: dict[int, set[str]] = defaultdict(set)

    for _, row in user_features_base.iterrows():
        user_id = int(row[user_col])

        if include_gender and "gender" in row.index:
            _add_token(result, user_id, "gender", row["gender"])

        if include_occupation and "occupation" in row.index:
            _add_token(result, user_id, "occupation", row["occupation"])

        if include_age_group and "age_group_id" in row.index:
            _add_token(result, user_id, "age_group", row["age_group_id"])

    return {entity_id: sorted(tokens) for entity_id, tokens in result.items()}


def build_item_feature_tokens(
    movie_genres: pd.DataFrame,
    movie_actors: pd.DataFrame,
    movie_directors: pd.DataFrame,
    movie_countries: pd.DataFrame,
) -> dict[int, list[str]]:
    """
    Build item feature tokens for LightFM using only relational tables.
    """

    feature_map = {}

    # genres
    for row in movie_genres.itertuples(index=False):
        feature_map.setdefault(row.movie_id, []).append(f"genre_id:{row.genre_id}")

    # actors
    for row in movie_actors.itertuples(index=False):
        feature_map.setdefault(row.movie_id, []).append(f"actor_id:{row.actor_id}")

    # directors
    for row in movie_directors.itertuples(index=False):
        feature_map.setdefault(row.movie_id, []).append(f"director_id:{row.director_id}")

    # countries
    for row in movie_countries.itertuples(index=False):
        feature_map.setdefault(row.movie_id, []).append(f"country_id:{row.country_id}")

    return feature_map


def filter_feature_map(
    feature_map: Dict[int, List[str]],
    allowed_ids: Iterable[int],
) -> Dict[int, List[str]]:
    allowed = set(int(x) for x in allowed_ids)
    return {
        int(entity_id): tokens
        for entity_id, tokens in feature_map.items()
        if int(entity_id) in allowed
    }


def collect_all_feature_names(
    feature_map: Dict[int, List[str]],
) -> List[str]:
    feature_names = set()

    for tokens in feature_map.values():
        feature_names.update(tokens)

    return sorted(feature_names)


def to_lightfm_feature_tuples(
    feature_map: Dict[int, List[str]],
) -> List[tuple[int, list[str]]]:
    """
    Convert feature map to LightFM-compatible iterable.

    Output format:
        [
            (1, ["gender:M", "occupation:3"]),
            (2, ["gender:F", "occupation:7"]),
        ]
    """
    return [
        (int(entity_id), list(tokens))
        for entity_id, tokens in sorted(feature_map.items(), key=lambda x: x[0])
    ]


def summarize_feature_map(
    feature_map: Dict[int, List[str]],
) -> pd.Series:
    lengths = pd.Series([len(tokens) for tokens in feature_map.values()])

    if lengths.empty:
        return pd.Series({
            "entities": 0,
            "min_features": 0,
            "p25_features": 0.0,
            "median_features": 0.0,
            "p75_features": 0.0,
            "max_features": 0,
        })

    return pd.Series({
        "entities": int(lengths.shape[0]),
        "min_features": int(lengths.min()),
        "p25_features": float(lengths.quantile(0.25)),
        "median_features": float(lengths.median()),
        "p75_features": float(lengths.quantile(0.75)),
        "max_features": int(lengths.max()),
    })