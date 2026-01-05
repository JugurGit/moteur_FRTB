# frtb/girr.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from curves import ZeroCurve
from market import Market
from portfolio import BondTrade, SwapTrade
from utils import SCENARIOS, clip_corr, basel_scale, fmt_int, fmt_num, inter_bucket, low_med_high


# --- swap pricing (identique logique swap.py)
def swap_annuity(curve: ZeroCurve, T: int) -> float:
    return sum(curve.df(i) for i in range(1, T + 1))


def par_swap_rate(curve: ZeroCurve, T: int) -> float:
    A = swap_annuity(curve, T)
    return (1.0 - curve.df(T)) / A


def pv_swap(curve: ZeroCurve, N: float, T: int, K: float, receive_fixed: bool) -> float:
    A = swap_annuity(curve, T)
    S = par_swap_rate(curve, T)
    return N * (K - S) * A if receive_fixed else N * (S - K) * A


# --- bond pricing (identique logique bond.py)
def bond_cashflows(notional: float, coupon_rate: float, maturity: int) -> List[Tuple[float, float]]:
    cpn = notional * coupon_rate
    out = []
    for yr in range(1, maturity + 1):
        if yr < maturity:
            out.append((float(yr), cpn))
        else:
            out.append((float(yr), cpn + notional))
    return out


def pv_bond(curve: ZeroCurve, notional: float, coupon_rate: float, maturity: int) -> float:
    total = 0.0
    for t, cf in bond_cashflows(notional, coupon_rate, maturity):
        total += cf * curve.df(t)
    return total


@dataclass(frozen=True)
class GirrConfig:
    rw_by_tenor: Dict[float, float]
    specified_currency_reduction: bool
    bump_bp: float
    gamma_inter_ccy_med: float
    scenario_rule: str   # "lowmedhigh" or "basel_scale"
    rho_rule: str        # "exp_absdiff" or "basel_tenor"
    rho_param: float     # a (exp_absdiff) OR theta (basel_tenor)


def scenario_gamma(cfg: GirrConfig, scenario: str) -> float:
    if cfg.scenario_rule == "lowmedhigh":
        return low_med_high(cfg.gamma_inter_ccy_med)[scenario]
    if cfg.scenario_rule == "basel_scale":
        return clip_corr(basel_scale(cfg.gamma_inter_ccy_med, scenario))
    raise ValueError("scenario_rule must be lowmedhigh or basel_scale")


def rho_tenor_basel(Tk: float, Tl: float, theta: float) -> float:
    if Tk <= 0.0 or Tl <= 0.0:
        return 0.4
    return max(math.exp(-theta * abs(math.log(Tk / Tl))), 0.4)


def scenario_adjust_corr(cfg: GirrConfig, scenario: str, base_corr: float) -> float:
    if cfg.scenario_rule == "lowmedhigh":
        return clip_corr(low_med_high(base_corr)[scenario])
    if cfg.scenario_rule == "basel_scale":
        return clip_corr(basel_scale(base_corr, scenario))
    raise ValueError("scenario_rule must be lowmedhigh or basel_scale")


def corr_matrix_for_tenors(tenors: List[float], cfg: GirrConfig, scenario: str) -> List[List[float]]:
    n = len(tenors)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                base = 1.0
            else:
                if cfg.rho_rule == "exp_absdiff":
                    base = math.exp(-cfg.rho_param * abs(tenors[i] - tenors[j]))
                elif cfg.rho_rule == "basel_tenor":
                    base = rho_tenor_basel(tenors[i], tenors[j], theta=cfg.rho_param)
                else:
                    raise ValueError("rho_rule must be exp_absdiff or basel_tenor")

            out[i][j] = scenario_adjust_corr(cfg, scenario, base)
    return out


# ---- bump&reprice générique (swaps + bonds)

PrepareCtx = Callable[[Any, ZeroCurve], Any]
PVWithCtx = Callable[[Any, ZeroCurve, Any], float]
CCYFunc = Callable[[Any], str]
TradePrinter = Callable[[Any, float], None]


def bump_and_reprice_sens(
    trades: List[Any],
    mkt: Market,
    cfg: GirrConfig,
    prepare_ctx: PrepareCtx,
    pv_with_ctx: PVWithCtx,
    ccy_func: CCYFunc,
    printer: Optional[TradePrinter],
    verbose: bool,
) -> Dict[str, Dict[float, float]]:
    tenors_nodes = sorted(cfg.rw_by_tenor.keys())
    out: Dict[str, Dict[float, float]] = {}

    for tr in trades:
        ccy = ccy_func(tr)
        curve0 = mkt.curves[ccy]
        ctx = prepare_ctx(tr, curve0)
        PV0 = pv_with_ctx(tr, curve0, ctx)

        sens_ccy = {T: 0.0 for T in tenors_nodes}
        for Tk in tenors_nodes:
            PVb = pv_with_ctx(tr, curve0.bumped(Tk, cfg.bump_bp), ctx)
            s = (PVb - PV0) / cfg.bump_bp
            sens_ccy[Tk] += mkt.convert(s, ccy, mkt.reporting_ccy)

        out.setdefault(ccy, {T: 0.0 for T in tenors_nodes})
        for T in tenors_nodes:
            out[ccy][T] += sens_ccy[T]

        if verbose and printer is not None:
            printer(tr, mkt.convert(PV0, ccy, mkt.reporting_ccy))

    return out


# ---- delta SBM

def girr_delta_sbm(bucket_sens: Dict[str, Dict[float, float]], cfg: GirrConfig, verbose: bool) -> Dict[str, Any]:
    if not bucket_sens:
        return {"K_final": 0.0, "details": {}}

    tenors_nodes = sorted(cfg.rw_by_tenor.keys())

    rw = dict(cfg.rw_by_tenor)
    if cfg.specified_currency_reduction:
        for T in rw:
            rw[T] /= math.sqrt(2.0)

    WS: Dict[str, Dict[float, float]] = {}
    S: Dict[str, float] = {}
    for ccy, sens in bucket_sens.items():
        WS[ccy] = {T: rw[T] * sens[T] for T in tenors_nodes}
        S[ccy] = sum(WS[ccy].values())

    if verbose:
        print("\n[3] Weighted sensitivities WS_k = RW_k * s_k")
        for ccy in WS:
            print(f"\n  Bucket {ccy}:")
            for T in tenors_nodes:
                if abs(WS[ccy][T]) > 1e-6:
                    print(f"    T={T:>5}Y | RW={100*rw[T]:.3f}% | WS={fmt_num(WS[ccy][T],6)}")
            print(f"    S_{ccy}={fmt_num(S[ccy],6)}")

    Kb: Dict[str, Dict[str, float]] = {sc: {} for sc in SCENARIOS}

    if verbose:
        print("\n[4] Intra-bucket aggregation: K_b = sqrt(WS' rho WS)")

    for sc in SCENARIOS:
        for ccy in WS:
            ten_nz = [T for T in tenors_nodes if WS[ccy][T] != 0.0]
            if not ten_nz:
                Kb[sc][ccy] = 0.0
                continue

            rho = corr_matrix_for_tenors(ten_nz, cfg, scenario=sc)
            w = [WS[ccy][T] for T in ten_nz]
            val = sum(w[i] * rho[i][j] * w[j] for i in range(len(w)) for j in range(len(w)))
            Kb[sc][ccy] = math.sqrt(max(val, 0.0))

        if verbose:
            parts = " | ".join(f"{ccy}: {fmt_int(Kb[sc][ccy])}" for ccy in sorted(WS.keys()))
            print(f"  {sc.upper()} -> {parts}")

    if verbose:
        print("\n[5] Inter-bucket aggregation")

    totals: Dict[str, float] = {}
    for sc in SCENARIOS:
        g = scenario_gamma(cfg, sc)
        totals[sc] = inter_bucket(Kb[sc], S, g)
        if verbose:
            print(f"  {sc.upper()}: gamma={g:.3f} | K_total={fmt_int(totals[sc])}")

    K_final = max(totals.values())
    worst = max(totals, key=lambda k: totals[k])

    if verbose:
        print("\n[6] FINAL GIRR Delta capital")
        print(f"  >>> GIRR Delta capital = {fmt_int(K_final)} (worst={worst.upper()})")

    return {"K_final": K_final, "totals": totals, "WS": WS, "S": S, "Kb": Kb}


# ---- helpers prêts à l'emploi pour swaps & bonds (wrappers)

def sensitivities_swaps(trades: List[SwapTrade], mkt: Market, cfg: GirrConfig, verbose: bool) -> Dict[str, Dict[float, float]]:
    def prep_swap(tr: SwapTrade, curve0: ZeroCurve) -> float:
        return par_swap_rate(curve0, tr.maturity)  # K0

    def pv_swap_ctx(tr: SwapTrade, curve: ZeroCurve, K0: float) -> float:
        return pv_swap(curve, tr.notional, tr.maturity, K0, tr.receive_fixed)

    def print_swap(tr: SwapTrade, pv0_rep: float) -> None:
        print(f"\n  Swap {tr.name} ({tr.ccy}) | maturity={tr.maturity}Y | receive_fixed={tr.receive_fixed}")
        print(f"    PV0 ~ {pv0_rep:,.2f} {mkt.reporting_ccy}".replace(",", " ") + " (par swap)")

    return bump_and_reprice_sens(
        trades=trades, mkt=mkt, cfg=cfg,
        prepare_ctx=prep_swap, pv_with_ctx=pv_swap_ctx, ccy_func=lambda t: t.ccy,
        printer=print_swap, verbose=verbose
    )


def sensitivities_bonds(trades: List[BondTrade], mkt: Market, cfg: GirrConfig, verbose: bool) -> Dict[str, Dict[float, float]]:
    def prep_bond(tr: BondTrade, curve0: ZeroCurve) -> None:
        return None

    def pv_bond_ctx(tr: BondTrade, curve: ZeroCurve, _: None) -> float:
        return pv_bond(curve, tr.notional, tr.coupon_rate, tr.maturity)

    def print_bond(tr: BondTrade, pv0_rep: float) -> None:
        print(f"\n  Bond {tr.name} ({tr.ccy}) | maturity={tr.maturity}Y | PV0={pv0_rep:,.2f} {mkt.reporting_ccy}".replace(",", " "))

    return bump_and_reprice_sens(
        trades=trades, mkt=mkt, cfg=cfg,
        prepare_ctx=prep_bond, pv_with_ctx=pv_bond_ctx, ccy_func=lambda t: t.ccy,
        printer=print_bond, verbose=verbose
    )
