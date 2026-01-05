# ui_common.py
from __future__ import annotations

import contextlib
import csv
import io
import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple
import datetime
from history_db import init_db as history_init_db, insert_run as history_insert_run


import streamlit as st

# Imports de TON projet (mêmes noms que main.py)
from curves import ZeroCurve
from demo import (
    demo_equity_config,
    demo_girr_cfg_bonds,
    demo_girr_cfg_swaps,
    demo_market,
    demo_portfolio,
)
from engine import FRTBEngine
from market import Market
from portfolio import BondTrade, EquityCallTrade, Portfolio, SwapTrade
from utils import fmt_int, fmt_money


# ----------------------------
# UX / page config
# ----------------------------
def apply_page_config() -> None:
    st.set_page_config(
        page_title="FRTB SBM Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_pro_css() -> None:
    st.markdown(
        """
<style>
/* Spacing + cards */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
div[data-testid="stMetric"] { background: rgba(255,255,255,0.04); padding: 14px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.08); }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }
hr { opacity: .25; }
.small-note { opacity: .75; font-size: 0.92rem; }
</style>
""",
        unsafe_allow_html=True,
    )


# ----------------------------
# Session state init
# ----------------------------
def init_session_state() -> None:
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = demo_portfolio()

    if "market" not in st.session_state or "bond_override" not in st.session_state:
        mkt, bond_override = demo_market()
        st.session_state["market"] = mkt
        st.session_state["bond_override"] = bond_override

    if "equity_cfg" not in st.session_state:
        st.session_state["equity_cfg"] = demo_equity_config()

    if "girr_cfg_swaps" not in st.session_state:
        st.session_state["girr_cfg_swaps"] = demo_girr_cfg_swaps()

    if "girr_cfg_bonds" not in st.session_state:
        st.session_state["girr_cfg_bonds"] = demo_girr_cfg_bonds()

    # results
    st.session_state.setdefault("last_run", None)
    st.session_state.setdefault("last_logs", "")
    st.session_state.setdefault("last_run_error", "")
        # history db
    try:
        history_init_db()
    except Exception:
        # si l'environnement est read-only, on ne casse pas l'app
        pass



def summary_kpis() -> Dict[str, Any]:
    p: Portfolio = st.session_state["portfolio"]
    status = "—"
    if st.session_state.get("last_run_error"):
        status = "❌ erreur"
    elif st.session_state.get("last_run") is not None:
        status = "✅ ok"
    return {
        "n_eq": len(p.equity_calls),
        "n_sw": len(p.girr_swaps),
        "n_bo": len(p.girr_bonds),
        "last_run_status": status,
    }


# ----------------------------
# Conversions (Portfolio <-> rows)
# ----------------------------
def equity_rows_from_portfolio(p: Portfolio) -> List[Dict[str, Any]]:
    return [asdict(t) for t in p.equity_calls]


def swaps_rows_from_portfolio(p: Portfolio) -> List[Dict[str, Any]]:
    return [asdict(t) for t in p.girr_swaps]


def bonds_rows_from_portfolio(p: Portfolio) -> List[Dict[str, Any]]:
    return [asdict(t) for t in p.girr_bonds]


def portfolio_from_rows(
    eq_rows: List[Dict[str, Any]],
    sw_rows: List[Dict[str, Any]],
    bo_rows: List[Dict[str, Any]],
) -> Portfolio:
    eq: List[EquityCallTrade] = []
    for r in eq_rows:
        if not r or all(str(v).strip() == "" for v in r.values()):
            continue
        eq.append(
            EquityCallTrade(
                name=str(r["name"]),
                bucket=int(r["bucket"]),
                N=float(r["N"]),
                S0=float(r["S0"]),
                K=float(r["K"]),
                T=float(r["T"]),
                r=float(r["r"]),
                q_repo=float(r["q_repo"]),
                sigma=float(r["sigma"]),
            )
        )

    sw: List[SwapTrade] = []
    for r in sw_rows:
        if not r or all(str(v).strip() == "" for v in r.values()):
            continue
        sw.append(
            SwapTrade(
                name=str(r["name"]),
                ccy=str(r["ccy"]).strip().upper(),
                notional=float(r["notional"]),
                maturity=int(r["maturity"]),
                receive_fixed=bool(r["receive_fixed"]),
            )
        )

    bo: List[BondTrade] = []
    for r in bo_rows:
        if not r or all(str(v).strip() == "" for v in r.values()):
            continue
        bo.append(
            BondTrade(
                name=str(r["name"]),
                ccy=str(r["ccy"]).strip().upper(),
                notional=float(r["notional"]),
                coupon_rate=float(r["coupon_rate"]),
                maturity=int(r["maturity"]),
            )
        )

    return Portfolio(eq, sw, bo)


# ----------------------------
# CSV portfolio (upload/download)
# ----------------------------
def portfolio_to_csv(p: Portfolio) -> str:
    buf = io.StringIO()
    fieldnames = [
        "type",
        "name",
        "bucket",
        "N",
        "S0",
        "K",
        "T",
        "r",
        "q_repo",
        "sigma",
        "ccy",
        "notional",
        "maturity",
        "receive_fixed",
        "coupon_rate",
    ]
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()

    for t in p.equity_calls:
        w.writerow(
            {
                "type": "EQUITY_CALL",
                "name": t.name,
                "bucket": t.bucket,
                "N": t.N,
                "S0": t.S0,
                "K": t.K,
                "T": t.T,
                "r": t.r,
                "q_repo": t.q_repo,
                "sigma": t.sigma,
            }
        )

    for t in p.girr_swaps:
        w.writerow(
            {
                "type": "GIRR_SWAP",
                "name": t.name,
                "ccy": t.ccy,
                "notional": t.notional,
                "maturity": t.maturity,
                "receive_fixed": int(bool(t.receive_fixed)),
            }
        )

    for t in p.girr_bonds:
        w.writerow(
            {
                "type": "GIRR_BOND",
                "name": t.name,
                "ccy": t.ccy,
                "notional": t.notional,
                "coupon_rate": t.coupon_rate,
                "maturity": t.maturity,
            }
        )

    return buf.getvalue()


def portfolio_from_csv_text(text: str) -> Portfolio:
    eq: List[EquityCallTrade] = []
    sw: List[SwapTrade] = []
    bo: List[BondTrade] = []

    f = io.StringIO(text)
    r = csv.DictReader(f)
    for row in r:
        t = (row.get("type") or "").strip().upper()
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


# ----------------------------
# Market <-> rows
# ----------------------------
def curve_to_rows(curve: ZeroCurve) -> List[Dict[str, Any]]:
    return [{"tenor": float(T), "zero": float(z)} for T, z in zip(curve.tenors, curve.zeros)]


def rows_to_curve(rows: List[Dict[str, Any]]) -> ZeroCurve:
    clean = []
    for r in rows:
        if r is None:
            continue
        if "tenor" not in r or "zero" not in r:
            continue
        if str(r["tenor"]).strip() == "" or str(r["zero"]).strip() == "":
            continue
        clean.append((float(r["tenor"]), float(r["zero"])))

    if not clean:
        raise ValueError("Courbe vide")
    clean.sort(key=lambda x: x[0])

    ten = tuple(t for t, _ in clean)
    if any(ten[i] <= 0.0 for i in range(len(ten))):
        raise ValueError("Tenors doivent être > 0")
    if any(ten[i] >= ten[i + 1] for i in range(len(ten) - 1)):
        raise ValueError("Tenors doivent être strictement croissants")

    z = tuple(v for _, v in clean)
    return ZeroCurve(ten, z)


def fx_to_rows(fx: Dict[Tuple[str, str], float]) -> List[Dict[str, Any]]:
    rows = []
    for (f, t), v in sorted(fx.items()):
        rows.append({"from": f, "to": t, "rate": float(v)})
    return rows


def rows_to_fx(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for r in rows:
        f = str(r.get("from", "")).strip().upper()
        t = str(r.get("to", "")).strip().upper()
        if not f or not t:
            continue
        out[(f, t)] = float(r.get("rate"))
    return out


def update_market_curves(mkt: Market, curves_by_ccy: Dict[str, ZeroCurve], fx: Optional[Dict[Tuple[str, str], float]] = None) -> Market:
    new_fx = dict(mkt.fx) if fx is None else dict(fx)
    new_curves = dict(mkt.curves)
    for ccy, c in curves_by_ccy.items():
        new_curves[ccy] = c
    return Market(reporting_ccy=mkt.reporting_ccy, fx=new_fx, curves=new_curves)


# ----------------------------
# Engine run (capture prints)
# ----------------------------
def build_engine_from_state() -> FRTBEngine:
    return FRTBEngine(
        market=st.session_state["market"],
        equity_cfg=st.session_state["equity_cfg"],
        girr_cfg_swaps=st.session_state["girr_cfg_swaps"],
        girr_cfg_bonds=st.session_state["girr_cfg_bonds"],
    )


def run_engine(port: Portfolio, use_bond_override: bool, verbose: bool) -> Tuple[Dict[str, Any], str]:
    engine = build_engine_from_state()
    bond_override = st.session_state["bond_override"] if use_bond_override else None

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = engine.run(port=port, bond_curves_override=bond_override, verbose=verbose)
    logs = buf.getvalue()
    return res, logs


# ----------------------------
# Serialization for export
# ----------------------------
def to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return {k: to_jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return str(x)


def results_to_json(res: Dict[str, Any]) -> str:
    return json.dumps(to_jsonable(res), indent=2, ensure_ascii=False)


def pretty_capital_block(res: Dict[str, Any]) -> Dict[str, str]:
    # res = {"equity":..., "girr_swaps":..., "girr_bonds":...}
    K_eq = float(res.get("equity", {}).get("K_final", 0.0))
    K_sw = float(res.get("girr_swaps", {}).get("K_final", 0.0))
    K_bo = float(res.get("girr_bonds", {}).get("K_final", 0.0))
    return {
        "Equity": fmt_money(K_eq),
        "GIRR Swaps": fmt_money(K_sw),
        "GIRR Bonds": fmt_money(K_bo),
        "Total (somme)": fmt_money(K_eq + K_sw + K_bo),
    }


def save_run_history(
    *,
    port: Portfolio,
    res: Optional[Dict[str, Any]],
    logs: str,
    use_bond_override: bool,
    verbose: bool,
    status: str = "ok",
    error_txt: str = "",
) -> Optional[int]:
    """
    Enregistre un run (ok/erreur) en base SQLite.
    Retourne run_id si succès, sinon None.
    """
    try:
        # timestamps
        now_local = datetime.datetime.now().isoformat(timespec="seconds")
        now_utc = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

        # KPIs
        K_eq = float((res or {}).get("equity", {}).get("K_final", 0.0))
        K_sw = float((res or {}).get("girr_swaps", {}).get("K_final", 0.0))
        K_bo = float((res or {}).get("girr_bonds", {}).get("K_final", 0.0))
        K_total = K_eq + K_sw + K_bo

        # portfolio CSV (pratique pour download)
        port_csv = portfolio_to_csv(port)

        # snapshot JSON (état complet)
        snap = {
            "portfolio": to_jsonable(port),
            "market": to_jsonable(st.session_state["market"]),
            "equity_cfg": to_jsonable(st.session_state["equity_cfg"]),
            "girr_cfg_swaps": to_jsonable(st.session_state["girr_cfg_swaps"]),
            "girr_cfg_bonds": to_jsonable(st.session_state["girr_cfg_bonds"]),
            "bond_override_used": bool(use_bond_override),
            "verbose": bool(verbose),
            "bond_override": to_jsonable(st.session_state.get("bond_override")) if use_bond_override else None,
        }
        snap_json = json.dumps(snap, indent=2, ensure_ascii=False)

        # results JSON
        res_json = results_to_json(res or {})

        meta = {
            "use_bond_override": bool(use_bond_override),
            "verbose": bool(verbose),
        }
        meta_json = json.dumps(meta, indent=2, ensure_ascii=False)

        run_id = history_insert_run(
            {
                "created_at_utc": now_utc,
                "created_at_local": now_local,
                "status": "error" if status != "ok" else "ok",
                "k_eq": K_eq,
                "k_sw": K_sw,
                "k_bo": K_bo,
                "k_total": K_total,
                "portfolio_csv": port_csv,
                "snapshot_json": snap_json,
                "results_json": res_json,
                "logs_txt": logs or "",
                "meta_json": meta_json,
                "error_txt": error_txt or "",
            }
        )
        return run_id
    except Exception:
        return None
