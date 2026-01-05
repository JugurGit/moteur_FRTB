# frtb/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from curves import ZeroCurve
from market import Market
from portfolio import Portfolio
from equity import EquityConfig, equity_sbm
from girr import GirrConfig, girr_delta_sbm, sensitivities_bonds, sensitivities_swaps
from utils import fmt_money


@dataclass
class FRTBEngine:
    market: Market
    equity_cfg: EquityConfig
    girr_cfg_swaps: GirrConfig
    girr_cfg_bonds: GirrConfig

    def run(
        self,
        port: Portfolio,
        bond_curves_override: Optional[Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        # 1) Equity
        res_eq = equity_sbm(port.equity_calls, self.equity_cfg, verbose=verbose)

        # 2) GIRR swaps
        if port.girr_swaps:
            if verbose:
                print("\n" + "=" * 90)
                print("GIRR — Swaps — Delta (SBM)")
                print("=" * 90)
                print("[0] Bump&reprice sensitivities s_k sur z(Tk)")
            sens_sw = sensitivities_swaps(port.girr_swaps, self.market, self.girr_cfg_swaps, verbose=verbose)
            res_sw = girr_delta_sbm(sens_sw, self.girr_cfg_swaps, verbose=verbose)
        else:
            res_sw = {"K_final": 0.0}

        # 3) GIRR bonds (avec override curves éventuel)
        if port.girr_bonds:
            if verbose:
                print("\n" + "=" * 90)
                print("GIRR — Bonds — Delta (SBM)")
                print("=" * 90)
                print("[0] Bump&reprice sensitivities s_k sur z(Tk)")

            if bond_curves_override is not None:
                tenors, z_eur_bond, z_usd_bond = bond_curves_override
                old_eur = self.market.curves["EUR"]
                old_usd = self.market.curves["USD"]
                try:
                    self.market.curves["EUR"] = ZeroCurve(tenors, z_eur_bond)
                    self.market.curves["USD"] = ZeroCurve(tenors, z_usd_bond)
                    sens_bo = sensitivities_bonds(port.girr_bonds, self.market, self.girr_cfg_bonds, verbose=verbose)
                    res_bo = girr_delta_sbm(sens_bo, self.girr_cfg_bonds, verbose=verbose)
                finally:
                    self.market.curves["EUR"] = old_eur
                    self.market.curves["USD"] = old_usd
            else:
                sens_bo = sensitivities_bonds(port.girr_bonds, self.market, self.girr_cfg_bonds, verbose=verbose)
                res_bo = girr_delta_sbm(sens_bo, self.girr_cfg_bonds, verbose=verbose)
        else:
            res_bo = {"K_final": 0.0}

        # Résumé (somme simple)
        if verbose:
            print("\n" + "=" * 90)
            print("RÉSUMÉ (par risk class) — et somme simple")
            print("=" * 90)
            K_eq = float(res_eq["K_final"])
            K_sw = float(res_sw["K_final"])
            K_bo = float(res_bo["K_final"])
            print(f"  Equity     : {fmt_money(K_eq)}")
            print(f"  GIRR Swaps : {fmt_money(K_sw)}")
            print(f"  GIRR Bonds : {fmt_money(K_bo)}")
            print(f"\n  Total (somme) : {fmt_money(K_eq + K_sw + K_bo)}")
            print("=" * 90)

        return {"equity": res_eq, "girr_swaps": res_sw, "girr_bonds": res_bo}
