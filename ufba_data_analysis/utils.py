from typing import Iterable

import pandas as pd
import yaml


def get_conf(path: str) -> dict:
    with open(path) as conf_path:
        cfg = yaml.safe_load(conf_path)
    return cfg


def filter_with_config(
    df: pd.DataFrame, column: str, filter_configs: dict
) -> pd.Series:
    if column in filter_configs.keys():
        assert isinstance(filter_configs[column]["filter"], bool)
        assert isinstance(filter_configs[column]["regex"], bool)
        if filter_configs[column]["filter"]:
            mask = df[column].str.contains(
                filter_configs[column]["contains"],
                regex=filter_configs[column]["regex"],
            )
            if filter_configs[column]["negate"]:
                mask = ~mask
            df = df.loc[mask, :]
    return df


def get_transform_cols(
    df_columns: list, transform_configs: dict, type: str
) -> Iterable:
    for col in df_columns:
        if col in transform_configs.keys():
            if transform_configs[col]["type"] == type:
                yield col
