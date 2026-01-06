# frtb/market.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from curves import ZeroCurve


@dataclass(frozen=True)
class Market:
    """
    Snapshot de marché pour le moteur FRTB.

    Contient :
    - reporting_ccy : devise de reporting (ex: "EUR") 
    - fx           : table FX (from_ccy, to_ccy) -> taux de change
                    ex: {("USD","EUR"): 0.92, ("EUR","USD"): 1/0.92}
    - curves       : courbes de taux par devise
                    ex: {"EUR": ZeroCurve(...), "USD": ZeroCurve(...)}

    """
    reporting_ccy: str
    fx: Dict[Tuple[str, str], float]     
    curves: Dict[str, ZeroCurve]         

    def convert(self, amount: float, from_ccy: str, to_ccy: str) -> float:
        """
        Convertit un montant d'une devise vers une autre à l'aide de la table FX.
        """
        # Pas de conversion si même devise
        if from_ccy == to_ccy:
            return amount

        # On cherche le taux FX direct
        key = (from_ccy, to_ccy)
        if key not in self.fx:
            raise ValueError(f"FX manquant pour {from_ccy}->{to_ccy}")

        # Conversion simple : amount * FX(from->to)
        return amount * self.fx[key]
