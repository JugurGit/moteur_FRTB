# frtb/utils.py
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

SCENARIOS: Tuple[str, str, str] = ("low", "medium", "high")


def fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}{abs(x):,.2f}".replace(",", " ")


def fmt_num(x: float, nd: int = 6) -> str:
    return f"{x:,.{nd}f}".replace(",", " ")


def fmt_int(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def pct(x: float) -> str:
    return f"{100.0 * x:.6f}%"


def clip_corr(x: float) -> float:
    return max(min(x, 1.0), -1.0)


def low_med_high(base: float) -> Dict[str, float]:
    """
    Règle low/medium/high utilisée dans tes scripts equity/swap:
      low  = max(2*base - 1, 0.75*base)
      med  = base
      high = min(1.25*base, 1)
    """
    return {
        "low": max(2.0 * base - 1.0, 0.75 * base),
        "medium": base,
        "high": min(1.25 * base, 1.0),
    }


def basel_scale(base: float, scenario: str) -> float:
    """Règle 0.75 / 1 / 1.25 (bond.py)."""
    if scenario == "low":
        return 0.75 * base
    if scenario == "medium":
        return 1.00 * base
    if scenario == "high":
        return 1.25 * base
    raise ValueError("scenario must be low/medium/high")


def inter_bucket(Kb: Dict[Any, float], X: Dict[Any, float], gamma: float) -> float:
    """
    Formule générique inter-bucket:
      K = sqrt( sum_b Kb[b]^2 + sum_{i<j} 2*gamma*X[i]*X[j] )

    - Pour Delta/Vega: X = S_b
    - Pour Curvature:  X = K_b^curv (comme ton code)
    """
    keys = sorted(Kb.keys())
    base = sum(Kb[k] ** 2 for k in keys)
    cross = 0.0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cross += 2.0 * gamma * X[keys[i]] * X[keys[j]]
    return math.sqrt(max(base + cross, 0.0))
