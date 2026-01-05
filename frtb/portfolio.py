# frtb/portfolio.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List


@dataclass
class EquityCallTrade:
    name: str
    bucket: int
    N: float
    S0: float
    K: float
    T: float
    r: float
    q_repo: float
    sigma: float


@dataclass
class SwapTrade:
    name: str
    ccy: str
    notional: float
    maturity: int
    receive_fixed: bool  # True Receive-Fixed, False Pay-Fixed


@dataclass
class BondTrade:
    name: str
    ccy: str
    notional: float
    coupon_rate: float
    maturity: int


@dataclass
class Portfolio:
    equity_calls: List[EquityCallTrade]
    girr_swaps: List[SwapTrade]
    girr_bonds: List[BondTrade]


def load_portfolio_csv(path: str) -> Portfolio:
    eq: List[EquityCallTrade] = []
    sw: List[SwapTrade] = []
    bo: List[BondTrade] = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = (row.get("type") or "").strip().upper()
            if t == "EQUITY_CALL":
                eq.append(EquityCallTrade(
                    name=row["name"],
                    bucket=int(row["bucket"]),
                    N=float(row["N"]),
                    S0=float(row["S0"]),
                    K=float(row["K"]),
                    T=float(row["T"]),
                    r=float(row["r"]),
                    q_repo=float(row["q_repo"]),
                    sigma=float(row["sigma"]),
                ))
            elif t == "GIRR_SWAP":
                sw.append(SwapTrade(
                    name=row["name"],
                    ccy=row["ccy"].strip().upper(),
                    notional=float(row["notional"]),
                    maturity=int(row["maturity"]),
                    receive_fixed=bool(int(row["receive_fixed"])),
                ))
            elif t == "GIRR_BOND":
                bo.append(BondTrade(
                    name=row["name"],
                    ccy=row["ccy"].strip().upper(),
                    notional=float(row["notional"]),
                    coupon_rate=float(row["coupon_rate"]),
                    maturity=int(row["maturity"]),
                ))
            elif not t:
                continue
            else:
                raise ValueError(f"type inconnu: {t}")

    return Portfolio(eq, sw, bo)
