# frtb/market.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from curves import ZeroCurve


@dataclass(frozen=True)
class Market:
    reporting_ccy: str
    fx: Dict[Tuple[str, str], float]     # (from,to) -> rate
    curves: Dict[str, ZeroCurve]         # ccy -> curve

    def convert(self, amount: float, from_ccy: str, to_ccy: str) -> float:
        if from_ccy == to_ccy:
            return amount
        key = (from_ccy, to_ccy)
        if key not in self.fx:
            raise ValueError(f"FX manquant pour {from_ccy}->{to_ccy}")
        return amount * self.fx[key]
