# frtb/equity.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

from portfolio import EquityCallTrade
from utils import SCENARIOS, fmt_money, inter_bucket, low_med_high


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def bs_call_price_delta_vega(S: float, K: float, T: float, r: float, q: float, sigma: float):
    if T <= 0.0:
        raise ValueError("T doit être > 0")
    if sigma <= 0.0:
        raise ValueError("sigma doit être > 0")

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)

    price = S * df_q * norm_cdf(d1) - K * df_r * norm_cdf(d2)
    delta = df_q * norm_cdf(d1)
    vega = S * df_q * sqrtT * norm_pdf(d1)

    return price, delta, vega, d1, d2


@dataclass(frozen=True)
class EquityConfig:
    delta_rw: Dict[int, Dict[str, float]]   # {bucket: {"spot":..., "repo":..., "curv":...}}
    rho_spot_repo_med: float
    gamma_inter_bucket_med: float
    rw_vega: float


def equity_sbm(trades: List[EquityCallTrade], cfg: EquityConfig, verbose: bool = True) -> Dict[str, Any]:
    if not trades:
        return {"K_final": 0.0, "details": {}}

    if verbose:
        print("\n" + "=" * 90)
        print("EQUITY — FRTB SA / SBM (Delta + Vega + Curvature)")
        print("=" * 90)

    # ---- 1) Metrics par trade
    tmet: Dict[str, Dict[str, float]] = {}
    if verbose:
        print("\n[1] Pricing + Greeks")

    for t in trades:
        price, delta, vega, d1, d2 = bs_call_price_delta_vega(t.S0, t.K, t.T, t.r, t.q_repo, t.sigma)
        PV = t.N * price

        s_spot = t.N * t.S0 * delta
        s_repo = t.N * (-t.T * t.S0 * delta)
        VR = t.N * vega * t.sigma

        tmet[t.name] = {"bucket": float(t.bucket), "PV": PV, "s_spot": s_spot, "s_repo": s_repo, "VR": VR}

        if verbose:
            print(f"\n  - {t.name}")
            print(f"    d1={d1:.6f} | d2={d2:.6f} | PV={fmt_money(PV)}")
            print(f"    s_spot={fmt_money(s_spot)} | s_repo={fmt_money(s_repo)} | VR={fmt_money(VR)}")

    buckets = sorted({int(x["bucket"]) for x in tmet.values()})

    # ---- 2) Delta
    if verbose:
        print("\n" + "-" * 90)
        print("[2] DELTA (SBM)")
        print("-" * 90)

    rho_sr = low_med_high(cfg.rho_spot_repo_med)
    gamma_ib = low_med_high(cfg.gamma_inter_bucket_med)

    bucket_delta: Dict[int, Dict[str, float]] = {}
    for b in buckets:
        s_spot_b = sum(m["s_spot"] for m in tmet.values() if int(m["bucket"]) == b)
        s_repo_b = sum(m["s_repo"] for m in tmet.values() if int(m["bucket"]) == b)

        rw_spot = cfg.delta_rw[b]["spot"]
        rw_repo = cfg.delta_rw[b]["repo"]

        WS_spot = rw_spot * s_spot_b
        WS_repo = rw_repo * s_repo_b

        Kb = {}
        for sc in SCENARIOS:
            rho = rho_sr[sc]
            Kb[sc] = math.sqrt(max(WS_spot * WS_spot + WS_repo * WS_repo + 2.0 * rho * WS_spot * WS_repo, 0.0))

        S_b = WS_spot + WS_repo
        bucket_delta[b] = {
            "WS_spot": WS_spot, "WS_repo": WS_repo, "S": S_b,
            "K_low": Kb["low"], "K_med": Kb["medium"], "K_high": Kb["high"],
        }

        if verbose:
            print(f"\n  Bucket {b}:")
            print(f"    WS_spot={fmt_money(WS_spot)} | WS_repo={fmt_money(WS_repo)} | S_b={fmt_money(S_b)}")
            print(f"    K_b low/med/high = {fmt_money(Kb['low'])} / {fmt_money(Kb['medium'])} / {fmt_money(Kb['high'])}")

    Kd = {}
    X = {b: bucket_delta[b]["S"] for b in buckets}
    for sc in SCENARIOS:
        key = {"low": "K_low", "medium": "K_med", "high": "K_high"}[sc]
        Kb_sc = {b: bucket_delta[b][key] for b in buckets}
        Kd[sc] = inter_bucket(Kb_sc, X, gamma_ib[sc])

    if verbose:
        print("\n  Inter-bucket DELTA:")
        print(f"    gamma low/med/high = {gamma_ib['low']:.6f} / {gamma_ib['medium']:.6f} / {gamma_ib['high']:.6f}")
        print(f"    KΔ low/med/high = {fmt_money(Kd['low'])} / {fmt_money(Kd['medium'])} / {fmt_money(Kd['high'])}")

    # ---- 3) Vega
    if verbose:
        print("\n" + "-" * 90)
        print("[3] VEGA (SBM)")
        print("-" * 90)

    bucket_vega: Dict[int, Dict[str, float]] = {}
    for b in buckets:
        VR_b = sum(m["VR"] for m in tmet.values() if int(m["bucket"]) == b)
        WS_b = cfg.rw_vega * VR_b
        K_b = abs(WS_b)
        bucket_vega[b] = {"S": WS_b, "K": K_b}

        if verbose:
            print(f"\n  Bucket {b}: VR_b={fmt_money(VR_b)} | WS_b={fmt_money(WS_b)} | K_b={fmt_money(K_b)}")

    Kv = {}
    Xv = {b: bucket_vega[b]["S"] for b in buckets}
    for sc in SCENARIOS:
        Kb_sc = {b: bucket_vega[b]["K"] for b in buckets}
        Kv[sc] = inter_bucket(Kb_sc, Xv, gamma_ib[sc])

    if verbose:
        print("\n  Inter-bucket VEGA:")
        print(f"    Kν low/med/high = {fmt_money(Kv['low'])} / {fmt_money(Kv['medium'])} / {fmt_money(Kv['high'])}")

    # ---- 4) Curvature
    if verbose:
        print("\n" + "-" * 90)
        print("[4] CURVATURE (SBM)")
        print("-" * 90)

    gammaC = low_med_high(cfg.gamma_inter_bucket_med ** 2)
    bucket_curv = {b: 0.0 for b in buckets}

    for t in trades:
        b = t.bucket
        rwc = cfg.delta_rw[b]["curv"]

        price0, delta0, _, _, _ = bs_call_price_delta_vega(t.S0, t.K, t.T, t.r, t.q_repo, t.sigma)
        V0 = t.N * price0
        s_spot = t.N * t.S0 * delta0

        V_plus = t.N * bs_call_price_delta_vega(t.S0 * (1.0 + rwc), t.K, t.T, t.r, t.q_repo, t.sigma)[0]
        V_minus = t.N * bs_call_price_delta_vega(t.S0 * (1.0 - rwc), t.K, t.T, t.r, t.q_repo, t.sigma)[0]

        CVR_plus = -(V_plus - V0 - rwc * s_spot)
        CVR_minus = -(V_minus - V0 + rwc * s_spot)

        bucket_curv[b] = max(bucket_curv[b], max(CVR_plus, 0.0), max(CVR_minus, 0.0))

        if verbose:
            print(f"\n  {t.name}")
            print(f"    RW^C={rwc:.2f} | V0={fmt_money(V0)} | V+={fmt_money(V_plus)} | V-={fmt_money(V_minus)}")
            print(f"    s_spot={fmt_money(s_spot)} | CVR+={fmt_money(CVR_plus)} | CVR-={fmt_money(CVR_minus)}")
            print(f"    -> K_bucket_curv now = {fmt_money(bucket_curv[b])}")

    if verbose:
        print("\n  Curvature par bucket:")
        for b in buckets:
            print(f"    Bucket {b}: K_b^curv={fmt_money(bucket_curv[b])}")

    Kc = {}
    Xc = dict(bucket_curv)  # comme ton code: X = K_b^curv
    for sc in SCENARIOS:
        Kb_sc = dict(bucket_curv)
        Kc[sc] = inter_bucket(Kb_sc, Xc, gammaC[sc])

    if verbose:
        print("\n  Inter-bucket CURVATURE:")
        print(f"    gamma^C low/med/high = {gammaC['low']:.6f} / {gammaC['medium']:.6f} / {gammaC['high']:.6f}")
        print(f"    Kcurv low/med/high = {fmt_money(Kc['low'])} / {fmt_money(Kc['medium'])} / {fmt_money(Kc['high'])}")

    totals = {sc: Kd[sc] + Kv[sc] + Kc[sc] for sc in SCENARIOS}
    K_final = max(totals.values())
    worst = max(totals, key=lambda k: totals[k])

    if verbose:
        print("\n" + "=" * 90)
        print("[5] Capital Equity final = max(low, medium, high)")
        print("=" * 90)
        print(f"  Low   : {fmt_money(totals['low'])}")
        print(f"  Medium: {fmt_money(totals['medium'])}")
        print(f"  High  : {fmt_money(totals['high'])}")
        print(f"\n  >>> Equity capital final = {fmt_money(K_final)} (worst={worst.upper()})")

    return {
        "K_final": K_final,
        "totals": totals,
        "bucket_delta": bucket_delta,
        "bucket_vega": bucket_vega,
        "bucket_curv": bucket_curv,
    }
