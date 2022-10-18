import argparse

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from ufba_data_analysis.utils import get_conf


def read_sisu_interim(interim_file: str) -> pd.DataFrame:
    return pd.read_parquet(interim_file)


def iterative_imputer(df: pd.DataFrame, max_iter: int) -> pd.DataFrame:
    imputer = IterativeImputer(max_iter=max_iter)
    df_imp = imputer.fit_transform(df)
    return pd.DataFrame(data=np.round(df_imp), columns=df.columns)


def fit_factor_analysis(
    imputed_df: pd.DataFrame,
    n_factors: int,
    inverted: bool = True,
    normalized: bool = True,
) -> pd.DataFrame:
    fa = FactorAnalyzer(n_factors=n_factors)
    fa_scores = fa.fit_transform(imputed_df)
    if inverted:
        fa_scores = fa_scores * -1
    if normalized:
        fa_scores = (fa_scores - np.min(fa_scores)) / (
            np.max(fa_scores) - np.min(fa_scores)
        )
    return fa_scores


def get_sisu_with_nse(interim_file: str, drop_nse_cols: bool = True) -> pd.DataFrame:
    cfg = get_conf("./params.yaml")["factor_analysis"]
    sisu_interim = read_sisu_interim(interim_file=interim_file)
    sisu_no_dupls = sisu_interim.drop_duplicates(subset=["ano_enem", "cpf"])
    sisu_imp = iterative_imputer(
        df=sisu_no_dupls.loc[:, cfg["nse_cols"]], max_iter=cfg["max_iter"]
    )
    sisu_no_dupls.loc[:, "nse"] = fit_factor_analysis(
        imputed_df=sisu_imp, n_factors=cfg["n_factors"]
    )
    sisu_with_nse = sisu_interim.merge(
        sisu_no_dupls.loc[:, ["ano_enem", "cpf", "nse"]], how="left"
    )
    if drop_nse_cols:
        sisu_with_nse = sisu_with_nse.drop(cfg["nse_cols"], axis=1)
    return sisu_with_nse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str)
    parser.add_argument("-o", type=str)
    args = parser.parse_args()
    sisu_with_nse = get_sisu_with_nse(interim_file=args.i)
    sisu_with_nse.to_parquet(path=args.o, index=False)
