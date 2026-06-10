"""Statistical analysis (paper Section 2.10).

Functions for the between-condition comparisons reported in the paper:

- ``cohens_d`` / ``cohens_d_ci``     : standardised mean difference + 95% CI.
- ``mean_diff_ci``                   : difference in means + 95% CI.
- ``mixed_effects``                  : linear mixed-effects model with a random
                                       intercept for ``group`` (e.g. game), to
                                       account for rallies nested within games.
- ``chi_square_homogeneity``         : re-exported from tactical.markov.

The large-sample SE of Cohen's d is
    SE(d) = sqrt( (n1 + n2) / (n1 * n2) + d^2 / (2 * (n1 + n2 - 2)) ).
"""
from __future__ import annotations

import numpy as np

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None

from ..tactical.markov import chi_square_homogeneity  # noqa: F401  (re-export)


def cohens_d(x1, x2) -> float:
    """Cohen's d with pooled standard deviation (group1 - group2)."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    n1, n2 = len(x1), len(x2)
    s_pooled = np.sqrt(((n1 - 1) * x1.var(ddof=1) + (n2 - 1) * x2.var(ddof=1))
                       / (n1 + n2 - 2))
    return float((x1.mean() - x2.mean()) / s_pooled) if s_pooled > 0 else 0.0


def cohens_d_ci(x1, x2, conf: float = 0.95) -> dict:
    """Cohen's d and its 95% confidence interval (large-sample SE)."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    n1, n2 = len(x1), len(x2)
    d = cohens_d(x1, x2)
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2 - 2)))
    z = stats.norm.ppf(1 - (1 - conf) / 2) if stats else 1.96
    return {"d": d, "ci_low": d - z * se, "ci_high": d + z * se, "se": float(se)}


def mean_diff_ci(x1, x2, conf: float = 0.95) -> dict:
    """Difference in means (group1 - group2) with a Welch 95% CI and t-test."""
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    n1, n2 = len(x1), len(x2)
    diff = x1.mean() - x2.mean()
    se = np.sqrt(x1.var(ddof=1) / n1 + x2.var(ddof=1) / n2)
    if stats is not None:
        df = se ** 4 / ((x1.var(ddof=1) / n1) ** 2 / (n1 - 1)
                        + (x2.var(ddof=1) / n2) ** 2 / (n2 - 1))
        tcrit = stats.t.ppf(1 - (1 - conf) / 2, df)
        t, p = stats.ttest_ind(x1, x2, equal_var=False)
    else:  # pragma: no cover
        tcrit, t, p = 1.96, float("nan"), float("nan")
    return {"diff": float(diff), "ci_low": float(diff - tcrit * se),
            "ci_high": float(diff + tcrit * se), "t": float(t), "p": float(p)}


def mixed_effects(values, condition, group) -> dict:
    """Linear mixed-effects model: value ~ condition + (1 | group).

    ``condition`` is a 0/1 indicator (e.g. 1 = scoring, 0 = conceding); ``group``
    identifies the game so that rallies nested within a game share a random
    intercept. Returns the fixed-effect estimate for ``condition`` with a 95% CI.
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pandas + statsmodels are required for mixed_effects.") from exc

    df = pd.DataFrame({"value": np.asarray(values, float),
                       "condition": np.asarray(condition, float),
                       "group": np.asarray(group)})
    model = smf.mixedlm("value ~ condition", df, groups=df["group"])
    res = model.fit(method="lbfgs", disp=False)
    beta = float(res.params["condition"])
    ci = res.conf_int().loc["condition"].tolist()
    return {"beta": beta, "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "p": float(res.pvalues["condition"])}


__all__ = ["cohens_d", "cohens_d_ci", "mean_diff_ci", "mixed_effects",
           "chi_square_homogeneity"]
