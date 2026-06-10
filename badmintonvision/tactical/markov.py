"""First-order Markov stroke-sequence modelling (paper Section 2.7).

Stroke sequence dynamics are modelled as a first-order Markov chain over the
stroke states {Forehand, Backhand, Jump_Smash}. The transition matrix
P[i, j] = P(next = j | current = i) summarises the aggregate one-step
dependency structure of a rally set.

This first-order Markov *transition structure* is distinct from lag-sequential
analysis (see ``lag_sequential.py`` and Sackett, 1987; Anguera, 1990): the
Markov matrix describes one-step transition probabilities, whereas
lag-sequential analysis tests whether a behaviour follows another at a given lag
relative to chance using adjusted residuals.

Conditions (e.g. scoring vs conceding) are compared with a chi-square test of
homogeneity on the transition *count* matrices, reported with the full
statistics (chi-square, degrees of freedom, exact p-value).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from scipy.stats import chi2_contingency
except ImportError:  # pragma: no cover
    chi2_contingency = None

DEFAULT_STATES = ("Forehand", "Backhand", "Jump_Smash")


def transition_counts(sequences: Sequence[Sequence[str]],
                      states: Sequence[str] = DEFAULT_STATES) -> np.ndarray:
    """Count one-step transitions across a list of stroke sequences."""
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    counts = np.zeros((n, n), dtype=np.int64)
    for seq in sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            if a in idx and b in idx:
                counts[idx[a], idx[b]] += 1
    return counts


def transition_matrix(counts: np.ndarray) -> np.ndarray:
    """Row-normalise a transition-count matrix into probabilities."""
    counts = np.asarray(counts, dtype=np.float64)
    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(row_sums > 0, counts / row_sums, 0.0)
    return probs


def chi_square_homogeneity(counts_a: np.ndarray, counts_b: np.ndarray) -> dict:
    """Chi-square test of homogeneity between two transition-count matrices.

    Compares the per-row transition distributions of two conditions. Returns
    chi-square statistic, degrees of freedom and p-value (summed across the
    origin states, as reported per stroke in the paper).
    """
    if chi2_contingency is None:
        raise ImportError("scipy is required for the chi-square test.")
    counts_a = np.asarray(counts_a, dtype=np.float64)
    counts_b = np.asarray(counts_b, dtype=np.float64)
    chi2_total, df_total = 0.0, 0
    per_state = []
    for i in range(counts_a.shape[0]):
        table = np.vstack([counts_a[i], counts_b[i]])
        table = table[:, table.sum(axis=0) > 0]      # drop empty columns
        if table.shape[1] < 2 or table.sum() == 0:
            per_state.append({"state": i, "chi2": 0.0, "df": 0, "p": 1.0})
            continue
        chi2, p, dof, _ = chi2_contingency(table, correction=False)
        chi2_total += chi2
        df_total += dof
        per_state.append({"state": i, "chi2": float(chi2), "df": int(dof),
                          "p": float(p)})
    return {"chi2": float(chi2_total), "df": int(df_total), "per_state": per_state}


__all__ = [
    "transition_counts",
    "transition_matrix",
    "chi_square_homogeneity",
    "DEFAULT_STATES",
]
