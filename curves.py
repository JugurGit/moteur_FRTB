# frtb/curves.py
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Tuple


def lin_interp(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    if abs(x1 - x0) < 1e-15:
        return y0
    w = (x - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


@dataclass(frozen=True)
class ZeroCurve:
    tenors: Tuple[float, ...]
    zeros: Tuple[float, ...]

    def z(self, t: float) -> float:
        T = self.tenors
        Z = self.zeros

        if t <= T[0]:
            return Z[0]
        if t >= T[-1]:
            return Z[-1]

        j = bisect.bisect_left(T, t)
        if abs(T[j] - t) < 1e-15:
            return Z[j]
        i = j - 1
        return lin_interp(t, T[i], Z[i], T[j], Z[j])

    def df(self, t: float) -> float:
        return math.exp(-self.z(t) * t)

    def bumped(self, bump_t: float, bump_size: float = 1e-4) -> "ZeroCurve":
        idx = None
        for k, T in enumerate(self.tenors):
            if abs(T - bump_t) < 1e-12:
                idx = k
                break
        if idx is None:
            raise ValueError(f"bump_t={bump_t} is not an exact curve node")
        z = list(self.zeros)
        z[idx] += bump_size
        return ZeroCurve(self.tenors, tuple(z))
