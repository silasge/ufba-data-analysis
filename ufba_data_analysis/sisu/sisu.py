import argparse
from glob import glob
import re

from loguru import logger
import numpy as np
import pandas as pd

from ufba_data_analysis.utils import (
    filter_with_config,
    get_conf,
    get_transform_cols,
)


def get_sisu_raw_dir(sisu_raw_dir: str) -> list:
    return glob(f"{sisu_raw_dir}/*.zip")


def read_sisu_csv(sisu_raw_path: str) -> pd.DataFrame:
    sisu_year = re.search(r"\d{4}", sisu_raw_path)[0]
    sisu_edition = re.search(r"\d{5}", sisu_raw_path)[0]
    return pd.read_csv(sisu_raw_path, compression="zip").assign(
        ano_sisu=sisu_year, per_ingr=sisu_edition
    )


def rename_sub_sisu_columns(sisu_df: pd.DataFrame) -> pd.DataFrame:
    sub_sisu_df = (
        sisu_df.rename(lambda x: re.sub("^COD_", "CO_", x), axis="columns")
        .rename(lambda x: re.sub("^UF_", "SG_UF_", x), axis="columns")
        .rename(lambda x: re.sub("^NU_NOTA", "NOTA", x), axis="columns")
    )
    return sub_sisu_df


def rename_map_sisu_columns(
    sisu_df: pd.DataFrame, rename_columns_conf: dict
) -> pd.DataFrame:
    sisu_year = str(sisu_df["ano_sisu"].unique()[0])
    all_years_mapper = rename_columns_conf["cols_all_years"]
    if sisu_year == "2015":
        mapper = {**all_years_mapper, **rename_columns_conf["cols_2015"]}
    elif sisu_year != "2015":
        mapper = {**all_years_mapper, **rename_columns_conf["cols_not_2015"]}
        if sisu_year == "2018":
            mapper = {**mapper, **rename_columns_conf["cols_only_2018"]}
        elif sisu_year != "2018":
            mapper = {**mapper, **rename_columns_conf["cols_only_neither"]}
    renamed_sisu_df = sisu_df.rename(mapper, axis="columns")
    return renamed_sisu_df.rename(str.lower, axis="columns")


def rename_sisu_columns(
    sisu_df: pd.DataFrame, rename_columns_conf: dict
) -> pd.DataFrame:
    sisu_df = rename_sub_sisu_columns(sisu_df)
    sisu_df = rename_map_sisu_columns(sisu_df, rename_columns_conf)
    return sisu_df


def filter_sisu_columns(
    renamed_sisu_df: pd.DataFrame, filter_rows_conf: dict
) -> pd.DataFrame:
    for col in renamed_sisu_df.columns:
        renamed_sisu_df = filter_with_config(renamed_sisu_df, col, filter_rows_conf)
    return renamed_sisu_df


def create_sisu_nota_media_column(
    renamed_sisu_df: pd.DataFrame, nota_media_conf: list
) -> pd.DataFrame:
    renamed_sisu_df["nota_media_enem"] = renamed_sisu_df[nota_media_conf].apply(
        np.mean, axis=1
    )
    return renamed_sisu_df


def create_sisu_cotista_column(
    renamed_sisu_df: pd.DataFrame, cotista_conf: list
) -> pd.DataFrame:
    renamed_sisu_df["cotista"] = renamed_sisu_df[cotista_conf].apply(
        lambda x: 1 if any(x == "sim") else 0, axis=1
    )
    return renamed_sisu_df


def create_sisu_necessidade_especial_column(
    renamed_sisu_df: pd.DataFrame, necessidade_especial_conf: list
) -> pd.DataFrame:
    renamed_sisu_df["necessidade_especial"] = renamed_sisu_df[
        necessidade_especial_conf
    ].apply(lambda x: 1 if any(x == 1) else 0, axis=1)
    return renamed_sisu_df


def create_sisu_columns(
    renamed_sisu_df: pd.DataFrame, create_columns_conf: dict
) -> pd.DataFrame:
    renamed_sisu_df = create_sisu_nota_media_column(
        renamed_sisu_df, create_columns_conf["nota_media_enem"]
    )
    renamed_sisu_df = create_sisu_cotista_column(
        renamed_sisu_df, create_columns_conf["cotista"]
    )
    renamed_sisu_df = create_sisu_necessidade_especial_column(
        renamed_sisu_df, create_columns_conf["necessidade_especial"]
    )
    return renamed_sisu_df


def transform_sisu_cpf_column(renamed_sisu_df: pd.DataFrame):
    renamed_sisu_df["cpf"] = (
        renamed_sisu_df["cpf"].astype("str").str.pad(width=11, fillchar="0")
    )
    return renamed_sisu_df


def transform_sisu_year_dependent_columns(
    renamed_sisu_df: pd.DataFrame, transform_columns_conf: dict
) -> pd.DataFrame:
    enem_year = str(int(renamed_sisu_df["ano_enem"].unique()[0]))
    year_dep_cols = get_transform_cols(
        df_columns=renamed_sisu_df.columns,
        transform_configs=transform_columns_conf,
        type="year_dependent",
    )
    for col in year_dep_cols:
        if enem_year == 2014:
            renamed_sisu_df[col] = renamed_sisu_df[col].map(
                transform_columns_conf[col]["mapper"][2014]
            )
        elif enem_year != 2014:
            renamed_sisu_df[col] = renamed_sisu_df[col].map(
                transform_columns_conf[col]["mapper"]["else"]
            )
    return renamed_sisu_df


def transform_sisu_regex_dependent_columns(
    renamed_sisu_df: pd.DataFrame, transform_columns_conf: dict
) -> pd.DataFrame:
    regex_dep_cols = get_transform_cols(
        df_columns=renamed_sisu_df.columns,
        transform_configs=transform_columns_conf,
        type="regex",
    )
    for col in regex_dep_cols:
        for key, val in transform_columns_conf[col]["mapper"].items():
            renamed_sisu_df[col] = renamed_sisu_df[col].apply(
                lambda x: val if re.search(key, x) else x
            )
            renamed_sisu_df[col] = renamed_sisu_df[col].str.replace("abi - ", "")
    return renamed_sisu_df


def transform_sisu_normal_columns(
    renamed_sisu_df: pd.DataFrame, transform_columns_conf: dict
) -> pd.DataFrame:
    normal_cols = get_transform_cols(
        df_columns=renamed_sisu_df.columns,
        transform_configs=transform_columns_conf,
        type="normal",
    )
    for col in normal_cols:
        renamed_sisu_df[col] = renamed_sisu_df[col].map(
            transform_columns_conf[col]["mapper"]
        )
    return renamed_sisu_df


def transform_sisu_columns(
    renamed_sisu_df: pd.DataFrame, transform_columns_conf: dict
) -> pd.DataFrame:
    renamed_sisu_df = transform_sisu_cpf_column(renamed_sisu_df)
    renamed_sisu_df = transform_sisu_year_dependent_columns(
        renamed_sisu_df, transform_columns_conf
    )
    renamed_sisu_df = transform_sisu_regex_dependent_columns(
        renamed_sisu_df, transform_columns_conf
    )
    renamed_sisu_df = transform_sisu_normal_columns(
        renamed_sisu_df, transform_columns_conf
    )
    return renamed_sisu_df


def select_sisu_columns(
    transformed_sisu_df: pd.DataFrame, select_columns_conf: list
) -> pd.DataFrame:
    return transformed_sisu_df.loc[:, select_columns_conf]


def get_sisu_data(
    directory_path: str,
    cfg_path: str = "./conf/sisu/process1.yaml",
) -> pd.DataFrame:
    process_cfg = get_conf(cfg_path)
    files = get_sisu_raw_dir(directory_path)
    sisu_dfs = [None] * len(files)
    for i, f in enumerate(files):
        logger.info(f"Reading {f}...")
        sisu_df = read_sisu_csv(sisu_raw_path=f)
        renamed_sisu_df = rename_sisu_columns(
            sisu_df, process_cfg["cond_rename_columns"]
        )
        filtered_sisu_df = filter_sisu_columns(
            renamed_sisu_df, process_cfg["filter_rows"]
        )
        sisu_df_with_new_cols = create_sisu_columns(
            filtered_sisu_df, process_cfg["create_columns"]
        )
        transformed_sisu_df = transform_sisu_columns(
            sisu_df_with_new_cols, process_cfg["transform_columns"]
        )
        selected_sisu_df = select_sisu_columns(
            transformed_sisu_df, process_cfg["select_columns"]
        )
        sisu_dfs[i] = selected_sisu_df
        logger.info("Done!")
    return pd.concat(sisu_dfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str)
    parser.add_argument("-o", type=str)
    parser.add_argument("--cfg", type=str)
    args = parser.parse_args()
    sisu_df = get_sisu_data(directory_path=args.d, cfg_path=args.cfg)
    sisu_df.to_parquet(args.o, index=False)
