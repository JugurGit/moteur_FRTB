from __future__ import annotations

import ast
import io
import json
import zipfile
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from curves import ZeroCurve
from equity import EquityConfig
from girr import GirrConfig
from history_db import clear_all, delete_run, get_run, init_db, list_runs
from market import Market
from portfolio import BondTrade, EquityCallTrade, Portfolio, SwapTrade
from ui_common import apply_page_config, apply_pro_css, init_session_state

apply_page_config()
apply_pro_css()
init_session_state()
init_db()

st.title("🕘 Historique")
st.caption("Historique des runs enregistrés en base SQLite (KPIs + snapshots + outputs téléchargeables).")


# =========================================================
# Helpers — restore
# =========================================================
def _parse_fx_key(k: Any) -> Optional[Tuple[str, str]]:
    """
    Accepte:
      - tuple/list ("USD","EUR")
      - string "('USD','EUR')" (cas actuel via to_jsonable)
      - string "USD->EUR", "USD/EUR", "USD,EUR"
    """
    if isinstance(k, (tuple, list)) and len(k) == 2:
        return (str(k[0]).strip().upper(), str(k[1]).strip().upper())

    if not isinstance(k, str):
        return None

    s = k.strip()

    # cas "('USD', 'EUR')"
    if s.startswith("(") and s.endswith(")"):
        try:
            t = ast.literal_eval(s)
            if isinstance(t, tuple) and len(t) == 2:
                return (str(t[0]).strip().upper(), str(t[1]).strip().upper())
        except Exception:
            pass

    # fallback séparateurs
    for sep in ["->", "/", ",", "|"]:
        if sep in s:
            a, b = s.replace(" ", "").split(sep, 1)
            if a and b:
                return (a.upper(), b.upper())

    return None


def _restore_zero_curve(d: Dict[str, Any]) -> ZeroCurve:
    ten = d.get("tenors", d.get("tenor", d.get("T", [])))
    z = d.get("zeros", d.get("zero", d.get("Z", [])))
    ten_t = tuple(float(x) for x in ten)
    z_t = tuple(float(x) for x in z)
    return ZeroCurve(ten_t, z_t)


def _restore_market(d: Dict[str, Any]) -> Market:
    reporting_ccy = str(d.get("reporting_ccy", "EUR")).strip().upper()

    fx_in = d.get("fx", {}) or {}
    fx: Dict[Tuple[str, str], float] = {}
    if isinstance(fx_in, dict):
        for k, v in fx_in.items():
            kk = _parse_fx_key(k)
            if kk is None:
                continue
            fx[kk] = float(v)

    curves_in = d.get("curves", {}) or {}
    curves: Dict[str, ZeroCurve] = {}
    if isinstance(curves_in, dict):
        for ccy, cd in curves_in.items():
            curves[str(ccy).strip().upper()] = _restore_zero_curve(cd)

    return Market(reporting_ccy=reporting_ccy, fx=fx, curves=curves)


def _restore_portfolio(d: Dict[str, Any]) -> Portfolio:
    eq_rows = d.get("equity_calls", []) or []
    sw_rows = d.get("girr_swaps", []) or []
    bo_rows = d.get("girr_bonds", []) or []

    eq = []
    for r in eq_rows:
        if not isinstance(r, dict):
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

    sw = []
    for r in sw_rows:
        if not isinstance(r, dict):
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

    bo = []
    for r in bo_rows:
        if not isinstance(r, dict):
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


def _restore_equity_cfg(d: Dict[str, Any]) -> EquityConfig:
    delta_rw_in = d.get("delta_rw", {}) or {}
    delta_rw: Dict[int, Dict[str, float]] = {}
    if isinstance(delta_rw_in, dict):
        for bk, val in delta_rw_in.items():
            b = int(bk)  # clés potentiellement string
            delta_rw[b] = {
                "spot": float(val.get("spot", 0.0)),
                "repo": float(val.get("repo", 0.0)),
                "curv": float(val.get("curv", 0.0)),
            }

    return EquityConfig(
        delta_rw=delta_rw,
        rho_spot_repo_med=float(d.get("rho_spot_repo_med", 0.0)),
        gamma_inter_bucket_med=float(d.get("gamma_inter_bucket_med", 0.0)),
        rw_vega=float(d.get("rw_vega", 0.0)),
    )


def _restore_girr_cfg(d: Dict[str, Any]) -> GirrConfig:
    rw_in = d.get("rw_by_tenor", {}) or {}
    rw: Dict[float, float] = {}
    if isinstance(rw_in, dict):
        for tk, v in rw_in.items():
            rw[float(tk)] = float(v)

    return GirrConfig(
        rw_by_tenor=rw,
        specified_currency_reduction=bool(d.get("specified_currency_reduction", False)),
        bump_bp=float(d.get("bump_bp", 1e-4)),
        gamma_inter_ccy_med=float(d.get("gamma_inter_ccy_med", 0.0)),
        scenario_rule=str(d.get("scenario_rule", "lowmedhigh")),
        rho_rule=str(d.get("rho_rule", "exp_absdiff")),
        rho_param=float(d.get("rho_param", 0.0)),
    )


def _restore_bond_override(x: Any) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]]:
    """
    snapshot["bond_override"] est soit None, soit une liste de 3 listes:
      [tenors, z_eur_bond, z_usd_bond]
    On renvoie (tuple(tenors), tuple(z_eur), tuple(z_usd))
    """
    if x is None:
        return None
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        return None
    ten = tuple(float(v) for v in x[0])
    z1 = tuple(float(v) for v in x[1])
    z2 = tuple(float(v) for v in x[2])
    return (ten, z1, z2)


def restore_session_from_snapshot(snapshot_json: str) -> Tuple[bool, str]:
    try:
        snap = json.loads(snapshot_json or "{}")
        if not isinstance(snap, dict):
            return False, "Snapshot invalide (pas un objet JSON)."

        # 1) Portfolio
        if "portfolio" in snap:
            st.session_state["portfolio"] = _restore_portfolio(snap["portfolio"])

        # 2) Market
        if "market" in snap:
            st.session_state["market"] = _restore_market(snap["market"])

        # 3) Configs
        if "equity_cfg" in snap:
            st.session_state["equity_cfg"] = _restore_equity_cfg(snap["equity_cfg"])
        if "girr_cfg_swaps" in snap:
            st.session_state["girr_cfg_swaps"] = _restore_girr_cfg(snap["girr_cfg_swaps"])
        if "girr_cfg_bonds" in snap:
            st.session_state["girr_cfg_bonds"] = _restore_girr_cfg(snap["girr_cfg_bonds"])

        # 4) Bond override (si présent)
        if "bond_override" in snap:
            bo = _restore_bond_override(snap.get("bond_override"))
            if bo is not None:
                st.session_state["bond_override"] = bo

        # 5) Reset résultats (évite confusion)
        st.session_state["last_run"] = None
        st.session_state["last_logs"] = ""
        st.session_state["last_run_error"] = ""

        return True, "Session restaurée (portfolio + market + configs)."
    except Exception as e:
        return False, f"Impossible de restaurer: {e}"


# =========================================================
# Filters
# =========================================================
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    status = st.selectbox("Filtre status", ["Tous", "ok", "error"], index=0, key="hist_status")
with c2:
    limit = st.slider("Nombre de runs", min_value=20, max_value=500, value=200, step=20, key="hist_limit")
with c3:
    st.write("")

status_db = None if status == "Tous" else status
runs = list_runs(limit=int(limit), status=status_db)

if not runs:
    st.info("Aucun run dans l’historique pour l’instant. Lancez un run dans **Run & Results**.")
    st.stop()

df = pd.DataFrame(runs)
for c in ["k_eq", "k_sw", "k_bo", "k_total"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

ok_n = int((df["status"] == "ok").sum()) if "status" in df.columns else 0
err_n = int((df["status"] == "error").sum()) if "status" in df.columns else 0
last_ts = str(df.iloc[0]["created_at_local"]) if len(df) else "—"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Runs (filtrés)", f"{len(df)}")
m2.metric("OK", f"{ok_n}")
m3.metric("Errors", f"{err_n}")
m4.metric("Dernier run", last_ts)

with st.expander("Voir la table des runs", expanded=False):
    show = df[["id", "created_at_local", "status", "k_eq", "k_sw", "k_bo", "k_total"]].copy()
    show = show.rename(
        columns={
            "created_at_local": "time",
            "k_eq": "K_eq",
            "k_sw": "K_sw",
            "k_bo": "K_bo",
            "k_total": "K_total",
        }
    )
    st.dataframe(show, use_container_width=True, hide_index=True)

labels = [
    f"#{int(r['id'])} | {r['created_at_local']} | {r['status']} | Total={float(r.get('k_total') or 0.0):,.2f}".replace(",", " ")
    for r in runs
]
sel = st.selectbox("Sélectionner un run", options=list(range(len(runs))), format_func=lambda i: labels[i], key="hist_select")
run_id = int(runs[sel]["id"])

rec = get_run(run_id)
if not rec:
    st.error("Run introuvable (il a peut-être été supprimé).")
    st.stop()

st.divider()

# =========================================================
# Run details + Restore button
# =========================================================
left, right = st.columns([1, 2])

with left:
    st.subheader(f"Run #{run_id}")
    st.write(f"**Time (local)**: {rec.get('created_at_local')}")
    st.write(f"**Status**: `{rec.get('status')}`")

    st.metric("K_eq", f"{float(rec.get('k_eq') or 0.0):,.2f}".replace(",", " "))
    st.metric("K_sw", f"{float(rec.get('k_sw') or 0.0):,.2f}".replace(",", " "))
    st.metric("K_bo", f"{float(rec.get('k_bo') or 0.0):,.2f}".replace(",", " "))
    st.metric("K_total", f"{float(rec.get('k_total') or 0.0):,.2f}".replace(",", " "))

    if rec.get("error_txt"):
        st.error(rec["error_txt"])

    st.divider()

    # ✅ Restore button
    if st.button("♻️ Restaurer ce run (portfolio + market + configs)", type="primary", use_container_width=True, key="hist_restore_btn"):
        ok, msg = restore_session_from_snapshot(rec.get("snapshot_json") or "{}")
        if ok:
            st.success(msg)
            st.toast("Vous pouvez maintenant aller dans Portfolio / Market / Configs.", icon="✅")
            st.rerun()
        else:
            st.error(msg)

with right:
    st.subheader("Téléchargements")

    portfolio_csv = rec.get("portfolio_csv") or ""
    snapshot_json = rec.get("snapshot_json") or "{}"
    results_json = rec.get("results_json") or "{}"
    logs_txt = rec.get("logs_txt") or ""

    cA, cB, cC, cD = st.columns(4)
    cA.download_button("⬇️ portfolio.csv", data=portfolio_csv, file_name=f"run_{run_id}_portfolio.csv", mime="text/csv", use_container_width=True)
    cB.download_button("⬇️ snapshot.json", data=snapshot_json, file_name=f"run_{run_id}_snapshot.json", mime="application/json", use_container_width=True)
    cC.download_button("⬇️ results.json", data=results_json, file_name=f"run_{run_id}_results.json", mime="application/json", use_container_width=True)
    cD.download_button("⬇️ logs.txt", data=logs_txt, file_name=f"run_{run_id}_logs.txt", mime="text/plain", use_container_width=True)

    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("portfolio.csv", portfolio_csv)
        z.writestr("snapshot.json", snapshot_json)
        z.writestr("results.json", results_json)
        z.writestr("logs.txt", logs_txt)
    zbuf.seek(0)

    st.download_button(
        "⬇️ Tout télécharger (zip)",
        data=zbuf.getvalue(),
        file_name=f"run_{run_id}_all.zip",
        mime="application/zip",
        use_container_width=True,
    )

st.divider()

# =========================================================
# Viewers
# =========================================================
tab1, tab2, tab3 = st.tabs(["Snapshot", "Results", "Logs"])

with tab1:
    try:
        st.json(json.loads(rec.get("snapshot_json") or "{}"), expanded=False)
    except Exception:
        st.code(rec.get("snapshot_json") or "{}", language="json")

with tab2:
    try:
        st.json(json.loads(rec.get("results_json") or "{}"), expanded=False)
    except Exception:
        st.code(rec.get("results_json") or "{}", language="json")

with tab3:
    if rec.get("logs_txt"):
        st.code(rec["logs_txt"], language="text")
    else:
        st.info("Pas de logs enregistrés pour ce run.")

st.divider()

# =========================================================
# Delete / clear
# =========================================================
c1, c2 = st.columns(2)

with c1:
    if st.button("🗑️ Supprimer ce run", use_container_width=True, key="hist_delete_one"):
        delete_run(run_id)
        st.success(f"Run #{run_id} supprimé.")
        st.rerun()

with c2:
    confirm = st.checkbox("Je confirme la suppression de tout l’historique", key="hist_confirm_clear")
    if st.button("🔥 Tout supprimer", use_container_width=True, disabled=not confirm, key="hist_clear_all"):
        clear_all()
        st.success("Historique supprimé.")
        st.rerun()
