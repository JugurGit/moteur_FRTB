# frtb/demo.py
from __future__ import annotations

from typing import Tuple

from curves import ZeroCurve
from market import Market
from portfolio import BondTrade, EquityCallTrade, Portfolio, SwapTrade
from equity import EquityConfig
from girr import GirrConfig


def demo_market() -> Tuple[Market, Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]]:
    tenors = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0)

    z_eur_swap = (0.0320, 0.0300, 0.0270, 0.0240, 0.0230, 0.0240, 0.0260, 0.0270, 0.0280, 0.0290)
    z_usd_swap = (0.0520, 0.0500, 0.0470, 0.0440, 0.0420, 0.0410, 0.0420, 0.0430, 0.0440, 0.0450)

    z_eur_bond_map = {0.25: 0.0270, 0.5: 0.0260, 1.0: 0.0245, 2.0: 0.0235, 3.0: 0.0230, 5.0: 0.0235, 10.0: 0.0255, 15.0: 0.0265, 20.0: 0.0260, 30.0: 0.0250}
    z_usd_bond_map = {0.25: 0.0530, 0.5: 0.0520, 1.0: 0.0500, 2.0: 0.0460, 3.0: 0.0430, 5.0: 0.0410, 10.0: 0.0420, 15.0: 0.0430, 20.0: 0.0435, 30.0: 0.0440}
    z_eur_bond = tuple(z_eur_bond_map[T] for T in tenors)
    z_usd_bond = tuple(z_usd_bond_map[T] for T in tenors)

    curves = {
        "EUR": ZeroCurve(tenors, z_eur_swap),
        "USD": ZeroCurve(tenors, z_usd_swap),
    }
    fx = {("USD", "EUR"): 0.92, ("EUR", "USD"): 1.0 / 0.92}
    return Market(reporting_ccy="EUR", fx=fx, curves=curves), (tenors, z_eur_bond, z_usd_bond)


def demo_equity_config() -> EquityConfig:
    delta_rw = {
        7: {"spot": 0.40, "repo": 0.0040, "curv": 0.40},
        8: {"spot": 0.50, "repo": 0.0050, "curv": 0.50},
    }
    return EquityConfig(
        delta_rw=delta_rw,
        rho_spot_repo_med=0.999,
        gamma_inter_bucket_med=0.15,
        rw_vega=0.7778,
    )


def demo_girr_cfg_swaps() -> GirrConfig:
    rw = {0.25: 0.017, 0.5: 0.017, 1.0: 0.016, 2.0: 0.013, 3.0: 0.012, 5.0: 0.011, 10.0: 0.011, 15.0: 0.011, 20.0: 0.011, 30.0: 0.011}
    return GirrConfig(
        rw_by_tenor=rw,
        specified_currency_reduction=True,
        bump_bp=1e-4,
        gamma_inter_ccy_med=0.50,
        scenario_rule="lowmedhigh",
        rho_rule="exp_absdiff",
        rho_param=0.018,
    )


def demo_girr_cfg_bonds() -> GirrConfig:
    rw_list = [0.017, 0.017, 0.015, 0.014, 0.014, 0.015, 0.015, 0.016, 0.016, 0.016]
    ten = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    rw = {ten[i]: rw_list[i] for i in range(len(ten))}
    return GirrConfig(
        rw_by_tenor=rw,
        specified_currency_reduction=False,
        bump_bp=1e-4,
        gamma_inter_ccy_med=0.50,
        scenario_rule="basel_scale",
        rho_rule="basel_tenor",
        rho_param=0.03,
    )


def demo_portfolio() -> Portfolio:
    eq = [
        EquityCallTrade(
            name="Trade A (Short Call) – Bucket 8",
            bucket=8, N=-10_000, S0=100.0, K=100.0, T=1.0, r=0.02, q_repo=0.01, sigma=0.25
        ),
        EquityCallTrade(
            name="Trade B (Short Call) – Bucket 7",
            bucket=7, N=-20_000, S0=50.0, K=55.0, T=0.5, r=0.02, q_repo=0.005, sigma=0.30
        ),
    ]
    sw = [
        SwapTrade(name="EUR IRS 5Y (Rec-Fixed)", ccy="EUR", notional=100_000_000.0, maturity=5, receive_fixed=True),
        SwapTrade(name="USD IRS 3Y (Pay-Fixed)", ccy="USD", notional=100_000_000.0, maturity=3, receive_fixed=False),
    ]
    bo = [
        BondTrade(name="EUR 3Y bond", ccy="EUR", notional=100_000_000.0, coupon_rate=0.03, maturity=3),
        BondTrade(name="USD 5Y bond", ccy="USD", notional=100_000_000.0, coupon_rate=0.045, maturity=5),
    ]
    return Portfolio(eq, sw, bo)
