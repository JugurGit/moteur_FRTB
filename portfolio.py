# frtb/portfolio.py
from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List


@dataclass
class EquityCallTrade:
    """
    Trade Equity (option call européenne) utilisé dans la risk class Equity (SA/SBM).

    Champs :
    - name   : identifiant lisible du trade
    - bucket : bucket réglementaire Equity (nombre entier)
    - N      : quantité / notionnel 
    - S0     : spot initial
    - K      : strike
    - T      : maturité (en années)
    - r      : taux sans risque (pour le discounting)
    - q_repo : coût de portage / repo (dividend yield)
    - sigma  : volatilité (Black-Scholes)
    """
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
    """
    Trade GIRR de type swap vanilla (IRS) utilisé pour la partie GIRR Delta (SBM).

    - name         : identifiant du trade
    - ccy          : devise du swap 
    - notional     : notionnel du swap
    - maturity     : maturité (en années)
    - receive_fixed: sens du swap
        True  -> Receive-Fixed (on reçoit le fixe / paye le float)
        False -> Pay-Fixed     (on paye le fixe / reçoit le float)
    """
    name: str
    ccy: str
    notional: float
    maturity: int
    receive_fixed: bool  


@dataclass
class BondTrade:
    """
    Trade GIRR de type bond à cashflows simples (coupon fixe annuel + remboursement in fine).

    - name        : identifiant du bond
    - ccy         : devise du bond (bucket GIRR)
    - notional    : nominal
    - coupon_rate : coupon annuel (taux, ex: 0.03)
    - maturity    : maturité (en années, entier ici)
    """
    name: str
    ccy: str
    notional: float
    coupon_rate: float
    maturity: int


@dataclass
class Portfolio:
    """
    Portfolio "multi risk class" pour le moteur de démo.

    On sépare explicitement les listes par typologie :
    - equity_calls : trades Equity (options)
    - girr_swaps   : trades GIRR (swaps)
    - girr_bonds   : trades GIRR (bonds)

    """
    equity_calls: List[EquityCallTrade]
    girr_swaps: List[SwapTrade]
    girr_bonds: List[BondTrade]


def load_portfolio_csv(path: str) -> Portfolio:
    """
    Charge un portfolio depuis un CSV (une ligne = un trade).

    Format attendu (colonne "type") :
    - EQUITY_CALL
    - GIRR_SWAP
    - GIRR_BOND
    """
    # Listes de trades par sous-portfolio
    eq: List[EquityCallTrade] = []
    sw: List[SwapTrade] = []
    bo: List[BondTrade] = []

    # Lecture CSV avec DictReader (header -> dictionnaire par ligne)
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)

        for row in r:
            t = (row.get("type") or "").strip().upper()

            # -----------------------------
            # EQUITY_CALL
            # -----------------------------
            if t == "EQUITY_CALL":
                eq.append(
                    EquityCallTrade(
                        name=row["name"],
                        bucket=int(row["bucket"]),
                        N=float(row["N"]),
                        S0=float(row["S0"]),
                        K=float(row["K"]),
                        T=float(row["T"]),
                        r=float(row["r"]),
                        q_repo=float(row["q_repo"]),
                        sigma=float(row["sigma"]),
                    )
                )

            # -----------------------------
            # GIRR_SWAP
            # -----------------------------
            elif t == "GIRR_SWAP":
                sw.append(
                    SwapTrade(
                        name=row["name"],
                        ccy=row["ccy"].strip().upper(),
                        notional=float(row["notional"]),
                        maturity=int(row["maturity"]),
                        receive_fixed=bool(int(row["receive_fixed"])),
                    )
                )

            # -----------------------------
            # GIRR_BOND
            # -----------------------------
            elif t == "GIRR_BOND":
                bo.append(
                    BondTrade(
                        name=row["name"],
                        ccy=row["ccy"].strip().upper(),
                        notional=float(row["notional"]),
                        coupon_rate=float(row["coupon_rate"]),
                        maturity=int(row["maturity"]),
                    )
                )
            elif not t:
                continue
            else:
                raise ValueError(f"type inconnu: {t}")

    return Portfolio(eq, sw, bo)
