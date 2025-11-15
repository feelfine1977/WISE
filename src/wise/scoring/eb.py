import pandas as pd


def shrink_mean(series: pd.Series, global_mean: float, counts: pd.Series, k: float) -> pd.Series:
    """
    Empirical-Bayes shrinkage for slice means.

    shrunk_mu = (n * mu + k * global_mean) / (n + k)
    """
    return (counts * series + k * global_mean) / (counts + k)
