# frtb/girr.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from curves import ZeroCurve
from market import Market
from portfolio import BondTrade, SwapTrade
from utils import SCENARIOS, clip_corr, basel_scale, fmt_int, fmt_num, inter_bucket, low_med_high


# =============================================================================
# 1) Pricing (swap + bond)
# =============================================================================
# Objectif : avoir une mécanique simple et transparente pour :
# - obtenir des PV (PV0 et PV bumped)
# - calculer des sensibilités par bump&reprice (dPV/dz(Tk))


# --- swap pricing 
def swap_annuity(curve: ZeroCurve, T: int) -> float:
    """
    A(T) = somme des facteurs d'actualisation des paiements fixes.
    """
    return sum(curve.df(i) for i in range(1, T + 1))


def par_swap_rate(curve: ZeroCurve, T: int) -> float:
    """
    Taux swap par (par swap rate) :
      S(T) = (1 - DF(T)) / A(T)
    avec A(T) l'annuité (somme DF des dates fixes).
    """
    A = swap_annuity(curve, T)
    return (1.0 - curve.df(T)) / A


def pv_swap(curve: ZeroCurve, N: float, T: int, K: float, receive_fixed: bool) -> float:
    """
    PV d'un swap vanilla "payer/receiver" dans une approche très simple :
      PV ≈ N * (K - S) * A   si Receive-Fixed
      PV ≈ N * (S - K) * A   si Pay-Fixed

    où :
      - K est le taux fixe contractualisé
      - S est le taux par du marché sur maturité T
      - A est l'annuité (somme DF)
    """
    A = swap_annuity(curve, T)
    S = par_swap_rate(curve, T)
    return N * (K - S) * A if receive_fixed else N * (S - K) * A


# --- bond pricing 
def bond_cashflows(notional: float, coupon_rate: float, maturity: int) -> List[Tuple[float, float]]:
    """
    Génère les cashflows d'un bond à coupons annuels :
    - coupons chaque année
    - remboursement du principal à maturité
    """
    cpn = notional * coupon_rate
    out = []
    for yr in range(1, maturity + 1):
        if yr < maturity:
            out.append((float(yr), cpn))
        else:
            out.append((float(yr), cpn + notional))
    return out


def pv_bond(curve: ZeroCurve, notional: float, coupon_rate: float, maturity: int) -> float:
    """
    PV du bond = somme des cashflows actualisés sur la ZeroCurve.
    """
    total = 0.0
    for t, cf in bond_cashflows(notional, coupon_rate, maturity):
        total += cf * curve.df(t)
    return total


# =============================================================================
# 2) Configuration FRTB GIRR (Risk Weights + règles corrélations)
# =============================================================================

@dataclass(frozen=True)
class GirrConfig:
    """
    Paramétrage GIRR SA/SBM (delta) dans ce mini moteur.

    - rw_by_tenor : RW_k par nœud de tenor (ex: 0.5Y, 1Y, 2Y, ...)
    - specified_currency_reduction : réduction RW pour currencies "spécifiées"
      (dans le cadre FRTB SA, certaines monnaies ont un facteur sqrt(2) simplifié)
    - bump_bp : taille du bump appliqué sur le taux zéro z(Tk)
    - gamma_inter_ccy_med : corrélation inter-bucket (entre currencies), scénario medium
    - scenario_rule : comment on dérive low/med/high à partir d'une corrélation base
        * "lowmedhigh"  : règle custom (low_med_high)
        * "basel_scale" : règle Bâle (0.75 / 1 / 1.25)
    - rho_rule : corrélation intra-bucket (entre tenors d'une même currency bucket)
        * "exp_absdiff" : exp(-a*|Tk-Tl|)
        * "basel_tenor" : exp(-theta*|log(Tk/Tl)|) floored à 0.4
    - rho_param : paramètre "a" (exp_absdiff) ou "theta" (basel_tenor)
    """
    rw_by_tenor: Dict[float, float]
    specified_currency_reduction: bool
    bump_bp: float
    gamma_inter_ccy_med: float
    scenario_rule: str   
    rho_rule: str        
    rho_param: float    


def scenario_gamma(cfg: GirrConfig, scenario: str) -> float:
    """
    Gamma inter-bucket (entre currencies) en fonction du scénario low/med/high.
    """
    if cfg.scenario_rule == "lowmedhigh":
        return low_med_high(cfg.gamma_inter_ccy_med)[scenario]
    if cfg.scenario_rule == "basel_scale":
        return clip_corr(basel_scale(cfg.gamma_inter_ccy_med, scenario))
    raise ValueError("scenario_rule must be lowmedhigh or basel_scale")


def rho_tenor_basel(Tk: float, Tl: float, theta: float) -> float:
    """
    Corrélation "Bâle-like" entre tenors Tk et Tl :
      rho = max( exp(-theta * |log(Tk/Tl)|), 0.4 )

    - Le floor à 0.4 reflète une borne minimale.
    """
    if Tk <= 0.0 or Tl <= 0.0:
        return 0.4
    return max(math.exp(-theta * abs(math.log(Tk / Tl))), 0.4)


def scenario_adjust_corr(cfg: GirrConfig, scenario: str, base_corr: float) -> float:
    """
    Applique la règle scénario (low/med/high) à une corrélation "base".
    """
    if cfg.scenario_rule == "lowmedhigh":
        return clip_corr(low_med_high(base_corr)[scenario])
    if cfg.scenario_rule == "basel_scale":
        return clip_corr(basel_scale(base_corr, scenario))
    raise ValueError("scenario_rule must be lowmedhigh or basel_scale")


def corr_matrix_for_tenors(tenors: List[float], cfg: GirrConfig, scenario: str) -> List[List[float]]:
    """
    Construit la matrice de corrélation intra-bucket ρ (entre tenors) pour un scénario donné.

    - Diagonale = 1
    - Hors-diagonale :
        * exp_absdiff : exp(-a*|Tk-Tl|)
        * basel_tenor : rho_tenor_basel(Tk,Tl,theta)
    - Puis on applique l'ajustement scénario + clipping.
    """
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


# =============================================================================
# 3) Bump & Reprice : calcul générique des sensibilités s_k = dPV/dz(Tk)
# =============================================================================
# Ici on veut une fonction réutilisable pour plusieurs types de trades (swaps, bonds).

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
    """
    Calcule les sensibilités GIRR (par currency bucket et par tenor node) via bump&reprice.

    Sortie :
      out[ccy][Tk] = somme_{trades in ccy} dPV/dz(Tk) (exprimée dans reporting_ccy)

    Workflow (trade par trade) :
    1) PV0 avec la courbe "base"
    2) pour chaque node Tk :
         - on bump z(Tk) de bump_bp
         - on recalcule PVb
         - s_k = (PVb - PV0) / bump_bp
    3) conversion FX vers reporting_ccy
    4) agrégation par currency bucket
    """
    tenors_nodes = sorted(cfg.rw_by_tenor.keys())
    out: Dict[str, Dict[float, float]] = {}

    for tr in trades:
        # 1) Identifier la currency bucket du trade
        ccy = ccy_func(tr)
        curve0 = mkt.curves[ccy]

        ctx = prepare_ctx(tr, curve0)

        # 3) PV "base"
        PV0 = pv_with_ctx(tr, curve0, ctx)

        # 4) Calcul des sensibilités sur tous les nœuds de tenor
        sens_ccy = {T: 0.0 for T in tenors_nodes}
        for Tk in tenors_nodes:
            # PV bumpé : on perturbe z(Tk) et on reprice
            PVb = pv_with_ctx(tr, curve0.bumped(Tk, cfg.bump_bp), ctx)

            # dérivée numérique dPV/dz(Tk)
            s = (PVb - PV0) / cfg.bump_bp

            # conversion en reporting_ccy 
            sens_ccy[Tk] += mkt.convert(s, ccy, mkt.reporting_ccy)

        # 5) Agrégation dans la structure de sortie
        out.setdefault(ccy, {T: 0.0 for T in tenors_nodes})
        for T in tenors_nodes:
            out[ccy][T] += sens_ccy[T]

        # 6) Logs optionnels par trade 
        if verbose and printer is not None:
            printer(tr, mkt.convert(PV0, ccy, mkt.reporting_ccy))

    return out


# =============================================================================
# 4) GIRR Delta SBM : pondération RW + agrégation intra/inter bucket
# =============================================================================

def girr_delta_sbm(bucket_sens: Dict[str, Dict[float, float]], cfg: GirrConfig, verbose: bool) -> Dict[str, Any]:
    """
    Calcule le capital GIRR Delta selon un schéma SBM simplifié :

    Entrée :
      bucket_sens[ccy][Tk] = s_k (sensibilité non pondérée) par currency bucket et tenor node

    Étapes :
    (i)   appliquer RW_k (et la réduction specified_currency_reduction si demandé)
    (ii)  WS_k = RW_k * s_k
    (iii) intra-bucket : K_b = sqrt( WS' * rho * WS ) (rho dépend du scénario)
    (iv)  inter-bucket : agrégation des K_b entre currencies via inter_bucket(Kb, S, gamma)
          où S_ccy = somme_k WS_k (exposition agrégée bucket)

    Sortie :
      - K_final : max sur scénarios low/med/high
      - totals  : K_total par scénario
      - WS, S, Kb : objets intermédiaires
    """
    if not bucket_sens:
        return {"K_final": 0.0, "details": {}}

    tenors_nodes = sorted(cfg.rw_by_tenor.keys())

    # RW de base
    rw = dict(cfg.rw_by_tenor)

    # Option FRTB : réduction pour currencies "spécifiées"
    # Ici on applique RW / sqrt(2) .
    if cfg.specified_currency_reduction:
        for T in rw:
            rw[T] /= math.sqrt(2.0)

    # WS_k et S_bucket (par currency)
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

    # Kb[scenario][ccy] = K_b intra-bucket
    Kb: Dict[str, Dict[str, float]] = {sc: {} for sc in SCENARIOS}

    if verbose:
        print("\n[4] Intra-bucket aggregation: K_b = sqrt(WS' rho WS)")

    for sc in SCENARIOS:
        for ccy in WS:
            # On ne conserve que les tenors non-nuls pour éviter matrices inutiles
            ten_nz = [T for T in tenors_nodes if WS[ccy][T] != 0.0]
            if not ten_nz:
                Kb[sc][ccy] = 0.0
                continue

            # Matrice rho 
            rho = corr_matrix_for_tenors(ten_nz, cfg, scenario=sc)

            # vecteur de WS
            w = [WS[ccy][T] for T in ten_nz]

            # forme quadratique w' rho w
            val = sum(w[i] * rho[i][j] * w[j] for i in range(len(w)) for j in range(len(w)))
            Kb[sc][ccy] = math.sqrt(max(val, 0.0))

        if verbose:
            parts = " | ".join(f"{ccy}: {fmt_int(Kb[sc][ccy])}" for ccy in sorted(WS.keys()))
            print(f"  {sc.upper()} -> {parts}")

    if verbose:
        print("\n[5] Inter-bucket aggregation")

    # Agrégation inter-bucket entre currencies
    totals: Dict[str, float] = {}
    for sc in SCENARIOS:
        g = scenario_gamma(cfg, sc)
        totals[sc] = inter_bucket(Kb[sc], S, g)
        if verbose:
            print(f"  {sc.upper()}: gamma={g:.3f} | K_total={fmt_int(totals[sc])}")

    # Capital final = pire scénario
    K_final = max(totals.values())
    worst = max(totals, key=lambda k: totals[k])

    if verbose:
        print("\n[6] FINAL GIRR Delta capital")
        print(f"  >>> GIRR Delta capital = {fmt_int(K_final)} (worst={worst.upper()})")

    return {"K_final": K_final, "totals": totals, "WS": WS, "S": S, "Kb": Kb}


# =============================================================================
# 5) Wrappers "prêts à l'emploi" : swaps et bonds
# =============================================================================
# Ces fonctions adaptent bump_and_reprice_sens à chaque type d'instrument
# en définissant :
# - comment préparer le contexte
# - comment re-pricer


def sensitivities_swaps(trades: List[SwapTrade], mkt: Market, cfg: GirrConfig, verbose: bool) -> Dict[str, Dict[float, float]]:
    """
    Sensibilités GIRR (dPV/dz(Tk)) pour une liste de swaps.

    Choix pédagogique important :
    - On "fixe" K0 au par rate initial (prepare_ctx) pour avoir PV0 ~ 0 (par swap),
      ce qui isole la mécanique du bump&reprice.
    """
    def prep_swap(tr: SwapTrade, curve0: ZeroCurve) -> float:
        return par_swap_rate(curve0, tr.maturity)

    def pv_swap_ctx(tr: SwapTrade, curve: ZeroCurve, K0: float) -> float:
        # PV du swap en gardant K0 constant (on reprice vs la courbe bumpée)
        return pv_swap(curve, tr.notional, tr.maturity, K0, tr.receive_fixed)

    def print_swap(tr: SwapTrade, pv0_rep: float) -> None:
        # Impression lisible dans les logs de run 
        print(f"\n  Swap {tr.name} ({tr.ccy}) | maturity={tr.maturity}Y | receive_fixed={tr.receive_fixed}")
        print(f"    PV0 ~ {pv0_rep:,.2f} {mkt.reporting_ccy}".replace(",", " ") + " (par swap)")

    return bump_and_reprice_sens(
        trades=trades,
        mkt=mkt,
        cfg=cfg,
        prepare_ctx=prep_swap,
        pv_with_ctx=pv_swap_ctx,
        ccy_func=lambda t: t.ccy,
        printer=print_swap,
        verbose=verbose
    )


def sensitivities_bonds(trades: List[BondTrade], mkt: Market, cfg: GirrConfig, verbose: bool) -> Dict[str, Dict[float, float]]:
    """
    Sensibilités GIRR (dPV/dz(Tk)) pour une liste de bonds.
    """
    def prep_bond(tr: BondTrade, curve0: ZeroCurve) -> None:
        # Pas de paramètre à figer comme K0 (contrairement au swap)
        return None

    def pv_bond_ctx(tr: BondTrade, curve: ZeroCurve, _: None) -> float:
        return pv_bond(curve, tr.notional, tr.coupon_rate, tr.maturity)

    def print_bond(tr: BondTrade, pv0_rep: float) -> None:
        print(
            f"\n  Bond {tr.name} ({tr.ccy}) | maturity={tr.maturity}Y | PV0={pv0_rep:,.2f} {mkt.reporting_ccy}"
            .replace(",", " ")
        )

    return bump_and_reprice_sens(
        trades=trades,
        mkt=mkt,
        cfg=cfg,
        prepare_ctx=prep_bond,
        pv_with_ctx=pv_bond_ctx,
        ccy_func=lambda t: t.ccy,
        printer=print_bond,
        verbose=verbose
    )
