"""Confidence intervals for success rates.

Results are reported as k successes out of n episodes. With small n a bare
percentage overstates certainty: 0 of 12 attacks succeeding is still
consistent with a true rate of about 25%. Wilson score intervals behave well
at 0% and 100%, where the textbook normal interval collapses to zero width.
"""

import math

Z95 = 1.959964


def wilson(k, n, z=Z95):
    """95% Wilson score interval for k successes in n trials, as (low, high)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value for paired 0/1 outcomes.

    b = pairs where only the first condition passed, c = only the second.
    Pairs where both agree carry no information about the difference.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_diff_ci(both, first_only, second_only, neither):
    """Difference in success rate, second minus first, for paired 0/1 outcomes,
    with a 95% interval (Newcombe 1998, method 10: Wilson intervals with a
    correction for the correlation between the pairs). Returns (diff, low, high)."""
    n = both + first_only + second_only + neither
    if n == 0:
        return 0.0, 0.0, 0.0
    p1, p2 = (both + first_only) / n, (both + second_only) / n
    l1, u1 = wilson(both + first_only, n)
    l2, u2 = wilson(both + second_only, n)
    den = math.sqrt((both + first_only) * (second_only + neither) * (both + second_only) * (first_only + neither))
    phi = (both * neither - first_only * second_only) / den if den else 0.0
    phi = max(phi, 0.0)   # a negative correlation is treated as 0: wider, conservative intervals
    d = p2 - p1
    low = d - math.sqrt(max((p2 - l2) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1) + (u1 - p1) ** 2, 0))
    high = d + math.sqrt(max((u2 - p2) ** 2 - 2 * phi * (u2 - p2) * (p1 - l1) + (p1 - l1) ** 2, 0))
    return d, max(low, -1.0), min(high, 1.0)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def pass_hat_k(successes, trials, k):
    """Unbiased estimate of pass^k for one task: the chance that k attempts
    drawn from its trials all succeed (tau-bench). Average it over tasks."""
    if trials < k:
        return None
    return math.comb(successes, k) / math.comb(trials, k)


def rate_ci(flags):
    """'83% [76–88] (n=150)' for a list of 0/1 outcomes, or '–' if empty."""
    flags = list(flags)
    if not flags:
        return "–"
    k, n = sum(flags), len(flags)
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f}% [{100 * lo:.0f}–{100 * hi:.0f}] (n={n})"
