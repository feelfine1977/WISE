from typing import Tuple
import numpy as np
import pandas as pd


def bootstrap_ci(
    values: pd.Series,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Simple bootstrap confidence interval for the mean.
    """
    rng = np.random.default_rng(random_state)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    samples = [values.sample(n=n, replace=True).mean() for _ in range(n_bootstrap)]
    lower = np.quantile(samples, alpha / 2)
    upper = np.quantile(samples, 1 - alpha / 2)
    return float(lower), float(upper)
