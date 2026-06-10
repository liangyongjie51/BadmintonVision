"""Lag-sequential analysis (paper Section 2.7; Sackett, 1987; Anguera, 1990).

Complements the first-order Markov description by testing, for each ordered
stroke pair (given -> target) at lag 1, whether the target follows the given
behaviour more or less often than expected by chance. The standard statistic is
the *adjusted residual*

    z_ij = (O_ij - E_ij) / sqrt( E_ij * (1 - p_i.) * (1 - p_.j) ),

where O_ij are observed lag-1 co-occurrence counts, E_ij = (row_i * col_j) / N
are the expected counts under independence, and p_i. , p_.j are the row/column
marginal proportions. |z| > 1.96 corresponds to p < 0.05 (two-tailed).

This is the analysis plotted as adjusted-residual heat-maps in the revised
manuscript (Figure 8g), corroborating the Markov-based description through an
independent, observational-analysis lens.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .markov import DEFAULT_STATES, transition_counts


def lag_sequential_residuals(
    sequences: Sequence[Sequence[str]],
    states: Sequence[str] = DEFAULT_STATES,
    lag: int = 1,
) -> dict:
    """Return observed, expected and adjusted-residual matrices at the given lag.

    For lag = 1 the observed matrix equals the Markov transition-count matrix.
    """
    if lag == 1:
        observed = transition_counts(sequences, states).astype(np.float64)
    else:
        idx = {s: i for i, s in enumerate(states)}
        n = len(states)
        observed = np.zeros((n, n), dtype=np.float64)
        for seq in sequences:
            for a, b in zip(seq[:-lag], seq[lag:]):
                if a in idx and b in idx:
                    observed[idx[a], idx[b]] += 1

    total = observed.sum()
    if total == 0:
        z = np.zeros_like(observed)
        return {"observed": observed, "expected": observed.copy(),
                "adjusted_residuals": z, "states": list(states)}

    row = observed.sum(axis=1, keepdims=True)        # given marginals
    col = observed.sum(axis=0, keepdims=True)        # target marginals
    expected = (row @ col) / total
    p_row = (row / total)                            # p_i.
    p_col = (col / total)                            # p_.j
    denom = np.sqrt(expected * (1 - p_row) * (1 - p_col))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(denom > 0, (observed - expected) / denom, 0.0)
    return {
        "observed": observed,
        "expected": expected,
        "adjusted_residuals": z,
        "states": list(states),
        "significant": np.abs(z) > 1.96,             # p < 0.05, two-tailed
    }


__all__ = ["lag_sequential_residuals"]
