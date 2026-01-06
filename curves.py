# frtb/curves.py
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Tuple


def lin_interp(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """
    Interpolation linéaire simple entre deux points (x0, y0) et (x1, y1).

    - On renvoie y(x) = y0 + w * (y1 - y0) avec w = (x - x0) / (x1 - x0).
    """
    
    if abs(x1 - x0) < 1e-15:
        return y0

    # Poids d'interpolation
    w = (x - x0) / (x1 - x0)
    return y0 + w * (y1 - y0)


@dataclass(frozen=True)
class ZeroCurve:
    """
    Courbe de taux zéro (zéro-coupon) définie par des nœuds (tenors) et leurs taux z(t).

    - tenors : maturités en années (ex: 0.25, 0.5, 1, 2, ...)
    - zeros  : taux zéro correspondants 

    """
    tenors: Tuple[float, ...]
    zeros: Tuple[float, ...]

    def z(self, t: float) -> float:
        """
        Renvoie le taux zéro z(t) à la maturité t.

        - Extrapolation plate aux extrémités (si t est en dehors des nœuds).
        - Interpolation linéaire entre deux nœuds sinon.
        """
        T = self.tenors
        Z = self.zeros

        # Extrapolation à gauche : on renvoie la première valeur
        if t <= T[0]:
            return Z[0]

        # Extrapolation à droite : on renvoie la dernière valeur
        if t >= T[-1]:
            return Z[-1]

        # Localiser l'intervalle [T[i], T[j]] qui encadre t (avec j = premier indice tel que T[j] >= t)
        j = bisect.bisect_left(T, t)

        # Si t tombe exactement sur un nœud, on renvoie la valeur du nœud
        if abs(T[j] - t) < 1e-15:
            return Z[j]

        # Sinon, interpolation entre le nœud précédent i=j-1 et le nœud j
        i = j - 1
        return lin_interp(t, T[i], Z[i], T[j], Z[j])

    def df(self, t: float) -> float:
        """
        Facteur d'actualisation P(0,t) obtenu à partir du taux zéro z(t).

        Hypothèse : taux zéro continûment composés
            df(t) = exp(-z(t) * t)
        """
        return math.exp(-self.z(t) * t)

    def bumped(self, bump_t: float, bump_size: float = 1e-4) -> "ZeroCurve":
        """
        Renvoie une nouvelle courbe où l'on a appliqué un bump (choc) sur UN nœud précis.

        - bump_t doit correspondre exactement à un tenor existant.
        - bump_size est typiquement un choc en taux (ex: 1e-4 = 1bp).

        Usage : bump&reprice pour approximer des sensibilités
            s_k ≈ (PV(z bumped at T_k) - PV(z)) / bump_size
        """
        idx = None

        for k, T in enumerate(self.tenors):
            if abs(T - bump_t) < 1e-12:
                idx = k
                break

        if idx is None:
            raise ValueError(f"bump_t={bump_t} is not an exact curve node")

        z = list(self.zeros)
        z[idx] += bump_size

        # On renvoie une nouvelle instance avec les mêmes tenors et les z modifiés
        return ZeroCurve(self.tenors, tuple(z))
