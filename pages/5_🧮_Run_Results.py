from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from ui_common import (
    apply_page_config,
    apply_pro_css,
    init_session_state,
    pretty_capital_block,
    run_engine,
    save_run_history,
)

# (optionnel) pour afficher corr matrix de manière fidèle au modèle
from girr import corr_matrix_for_tenors

apply_page_config()
apply_pro_css()
init_session_state()

# ---- petite couche CSS dédiée à cette page (cards/steps)
st.markdown(
    """
<style>
.step-card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  padding: 14px 16px;
  border-radius: 16px;
  margin: 8px 0 12px 0;
}
.step-title { font-size: 1.05rem; font-weight: 650; margin-bottom: 6px; }
.muted { opacity: .75; }
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.15);
  background: rgba(255,255,255,0.05);
  font-size: 0.85rem;
  margin-left: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🧮 Run & Results")
st.caption("Exécute FRTBEngine.run() et affiche les calculs étape par étape (Equity + GIRR Swaps + GIRR Bonds).")

p = st.session_state["portfolio"]

# -----------------------------
# Controls (run)
# -----------------------------
c1, c2, c3 = st.columns(3)
with c1:
    verbose = st.checkbox("Verbose (capturer les prints)", value=True, key="rr_verbose")
with c2:
    use_override = st.checkbox(
        "Appliquer bond_curves_override (demo_market)",
        value=True,
        key="rr_use_override",
    )
with c3:
    show_logs_snippets = st.checkbox(
        "Afficher extraits de logs dans les steps",
        value=True,
        key="rr_show_snippets",
    )

if st.button("▶️ Lancer le run", type="primary", use_container_width=True, key="rr_run_btn"):
    try:
        res, logs = run_engine(port=p, use_bond_override=use_override, verbose=verbose)
        st.session_state["last_run"] = res
        st.session_state["last_logs"] = logs
        st.session_state["last_run_error"] = ""
        st.success("Run terminé ✅")
        run_id = save_run_history(
            port=p,
            res=res,
            logs=logs,
            use_bond_override=use_override,
            verbose=verbose,
            status="ok",
            error_txt="",
        )
        if run_id is not None:
            st.toast(f"Run enregistré (id={run_id})", icon="🗄️")

    except Exception as e:
        st.session_state["last_run"] = None
        st.session_state["last_logs"] = ""
        st.session_state["last_run_error"] = str(e)
        st.error(f"Erreur run: {e}")
        _ = save_run_history(
            port=p,
            res=None,
            logs="",
            use_bond_override=use_override,
            verbose=verbose,
            status="error",
            error_txt=str(e),
        )


if st.session_state.get("last_run_error"):
    st.error(st.session_state["last_run_error"])

res = st.session_state.get("last_run")
logs = st.session_state.get("last_logs", "")

if res is None:
    st.info("Aucun résultat pour l’instant. Cliquez sur **Lancer le run**.")
    st.stop()

st.divider()

# -----------------------------
# Summary metrics
# -----------------------------
cap = pretty_capital_block(res)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Equity", cap["Equity"])
m2.metric("GIRR Swaps", cap["GIRR Swaps"])
m3.metric("GIRR Bonds", cap["GIRR Bonds"])
m4.metric("Total (somme)", cap["Total (somme)"])

st.divider()

# -----------------------------
# Helpers
# -----------------------------
def _step_box(title: str, subtitle: Optional[str] = None):
    st.markdown('<div class="step-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="step-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="muted">{subtitle}</div>', unsafe_allow_html=True)


def _step_box_end():
    st.markdown("</div>", unsafe_allow_html=True)


def _extract_between(text: str, start: str, end: Optional[str] = None) -> str:
    if not text:
        return ""
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start)) if end else -1
    if j < 0:
        return text[i:]
    return text[i:j]


def _worst_scenario(totals: Dict[str, float]) -> str:
    if not totals:
        return "—"
    k = max(totals, key=lambda s: float(totals[s]))
    return k.upper()


def _df_money(df: pd.DataFrame, cols: List[str], nd: int = 2) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(nd)
    return out


def _compute_girr_sens_from_ws(ws: Dict[str, Dict[float, float]], cfg, tenors: List[float]) -> Dict[str, Dict[float, float]]:
    """
    Reconstruit s_k approximativement via s_k = WS_k / RW_k,
    en tenant compte de specified_currency_reduction (RW / sqrt(2)).
    """
    if not ws:
        return {}
    rw = dict(cfg.rw_by_tenor)
    if cfg.specified_currency_reduction:
        for T in list(rw.keys()):
            rw[T] = rw[T] / math.sqrt(2.0)

    sens: Dict[str, Dict[float, float]] = {}
    for ccy, ws_ccy in ws.items():
        sens[ccy] = {}
        for T in tenors:
            rwT = rw.get(float(T))
            if rwT is None or abs(rwT) < 1e-18:
                sens[ccy][float(T)] = 0.0
            else:
                sens[ccy][float(T)] = float(ws_ccy.get(T, 0.0)) / float(rwT)
    return sens


def _ws_table(ws: Dict[str, Dict[float, float]]) -> pd.DataFrame:
    rows = []
    for ccy, d in ws.items():
        for T, v in d.items():
            rows.append({"ccy": ccy, "tenor": float(T), "WS": float(v)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["ccy", "tenor"]).reset_index(drop=True)
    return df


def _plot_ws(ws: Dict[str, Dict[float, float]], title: str):
    df = _ws_table(ws)
    if df.empty:
        st.info("Aucune WS à afficher.")
        return
    st.caption(title)
    for ccy in sorted(df["ccy"].unique()):
        d = df[df["ccy"] == ccy][["tenor", "WS"]].set_index("tenor")
        st.line_chart(d, use_container_width=True)


# -----------------------------
# Tabs
# -----------------------------
tab_eq, tab_sw, tab_bo, tab_logs = st.tabs(["Equity (Steps)", "GIRR Swaps (Steps)", "GIRR Bonds (Steps)", "Logs (Raw)"])

# =========================================================
# EQUITY
# =========================================================
with tab_eq:
    eq = res.get("equity", {}) or {}
    totals = eq.get("totals", {}) or {}
    worst = _worst_scenario(totals)

    cA, cB = st.columns([1, 2])
    with cA:
        st.metric("K_final", f"{float(eq.get('K_final', 0.0)):,.2f}".replace(",", " "))
        st.markdown(f'Worst scenario <span class="badge">{worst}</span>', unsafe_allow_html=True)
    with cB:
        if totals:
            st.write("Totaux par scénario:")
            st.dataframe(pd.DataFrame([totals]).rename_axis("scenario", axis=1), use_container_width=True)

    st.divider()

    # Step 1 — Pricing + Greeks (trade-level) => via logs
    _step_box("① Pricing + Greeks (par trade)", "PV, d1/d2, s_spot, s_repo, VR (issu des logs).")
    if show_logs_snippets and logs:
        snippet = _extract_between(logs, "[1] Pricing + Greeks", "[2] DELTA (SBM)")
        if snippet.strip():
            st.code(snippet.strip(), language="text")
        else:
            st.info("Aucun extrait trouvé (peut arriver si verbose=False).")
    else:
        st.info("Activez 'Verbose' et relancez pour voir le détail par trade.")
    _step_box_end()

    # Step 2 — Delta (bucket + interbucket)
    bd = eq.get("bucket_delta", {}) or {}
    _step_box("② DELTA (SBM)", "Weighted Sensitivities (WS_spot/WS_repo), K_b (low/med/high) et agrégation inter-bucket.")
    if bd:
        df = pd.DataFrame(
            [
                {
                    "bucket": int(b),
                    "WS_spot": float(v.get("WS_spot", 0.0)),
                    "WS_repo": float(v.get("WS_repo", 0.0)),
                    "S_b": float(v.get("S", 0.0)),
                    "K_low": float(v.get("K_low", 0.0)),
                    "K_med": float(v.get("K_med", 0.0)),
                    "K_high": float(v.get("K_high", 0.0)),
                }
                for b, v in bd.items()
            ]
        ).sort_values("bucket")
        df = _df_money(df, ["WS_spot", "WS_repo", "S_b", "K_low", "K_med", "K_high"], nd=2)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # mini chart: WS by bucket
        chart = df[["bucket", "WS_spot", "WS_repo"]].set_index("bucket")
        st.caption("WS par bucket (spot/repo)")
        st.bar_chart(chart, use_container_width=True)
    else:
        st.info("Pas de données delta.")
    _step_box_end()

    # Step 3 — Vega
    bv = eq.get("bucket_vega", {}) or {}
    _step_box("③ VEGA (SBM)", "VR par bucket, WS=RW*VR, K_b=|WS| et agrégation inter-bucket.")
    if bv:
        dfv = pd.DataFrame(
            [{"bucket": int(b), "S": float(v.get("S", 0.0)), "K": float(v.get("K", 0.0))} for b, v in bv.items()]
        ).sort_values("bucket")
        dfv = _df_money(dfv, ["S", "K"], nd=2)
        st.dataframe(dfv, use_container_width=True, hide_index=True)

        st.caption("WS vega par bucket")
        st.bar_chart(dfv[["bucket", "S"]].set_index("bucket"), use_container_width=True)
    else:
        st.info("Pas de données vega.")
    _step_box_end()

    # Step 4 — Curvature
    bc = eq.get("bucket_curv", {}) or {}
    _step_box("④ CURVATURE (SBM)", "K_b^curv par bucket puis agrégation inter-bucket (issu des logs).")
    if bc:
        dfc = pd.DataFrame([{"bucket": int(b), "K_curv": float(v)} for b, v in bc.items()]).sort_values("bucket")
        dfc = _df_money(dfc, ["K_curv"], nd=2)
        st.dataframe(dfc, use_container_width=True, hide_index=True)

        st.caption("K_curv par bucket")
        st.bar_chart(dfc.set_index("bucket"), use_container_width=True)
    else:
        st.info("Pas de données curvature.")
    if show_logs_snippets and logs:
        snippet = _extract_between(logs, "[4] CURVATURE (SBM)", "[5] Capital Equity final")
        if snippet.strip():
            with st.expander("Détail curvature (extrait des logs)", expanded=False):
                st.code(snippet.strip(), language="text")
    _step_box_end()

    # Step 5 — Final
    _step_box("⑤ Capital final Equity", "K_final = max( low, medium, high ).")
    if totals:
        dfT = pd.DataFrame([{"scenario": k, "K": float(v)} for k, v in totals.items()]).sort_values("K", ascending=False)
        st.dataframe(_df_money(dfT, ["K"], nd=2), use_container_width=True, hide_index=True)
    _step_box_end()

# =========================================================
# GIRR SWAPS
# =========================================================
with tab_sw:
    sw = res.get("girr_swaps", {}) or {}
    totals = sw.get("totals", {}) or {}
    worst = _worst_scenario(totals)

    cA, cB = st.columns([1, 2])
    with cA:
        st.metric("K_final", f"{float(sw.get('K_final', 0.0)):,.0f}".replace(",", " "))
        st.markdown(f'Worst scenario <span class="badge">{worst}</span>', unsafe_allow_html=True)
    with cB:
        if totals:
            st.write("Totaux par scénario:")
            st.dataframe(pd.DataFrame([totals]).rename_axis("scenario", axis=1), use_container_width=True)

    st.divider()

    ws = sw.get("WS", {}) or {}
    Kb = sw.get("Kb", {}) or {}
    cfg_sw = st.session_state["girr_cfg_swaps"]
    tenors = sorted(cfg_sw.rw_by_tenor.keys())

    # Step 0 — Bump & Reprice (logs)
    _step_box("⓪ Bump & Reprice (sensibilités)", "PV0 et mécanique bump (issu des logs).")
    if show_logs_snippets and logs:
        snippet = _extract_between(logs, "GIRR — Swaps — Delta (SBM)", "GIRR — Bonds — Delta (SBM)")
        if snippet.strip():
            st.code(snippet.strip(), language="text")
        else:
            st.info("Aucun extrait trouvé.")
    else:
        st.info("Activez 'Verbose' et relancez pour voir le détail bump&reprice.")
    _step_box_end()

    # Step 1 — Sensitivities (reconstruites)
    _step_box("① s_k (reconstruites)", "On reconstruit s_k ≈ WS_k / RW_k .")
    if ws:
        sens = _compute_girr_sens_from_ws(ws, cfg_sw, tenors)
        rows = []
        for ccy, d in sens.items():
            for T in tenors:
                rows.append({"ccy": ccy, "tenor": float(T), "s_k": float(d.get(float(T), 0.0))})
        df_s = pd.DataFrame(rows).sort_values(["ccy", "tenor"])
        df_s["s_k"] = df_s["s_k"].round(2)
        st.dataframe(df_s, use_container_width=True, hide_index=True)
    else:
        st.info("Pas de WS disponibles.")
    _step_box_end()

    # Step 2 — Weighted sensitivities
    _step_box("② Weighted sensitivities WS_k", "WS_k = RW_k * s_k. ")
    if ws:
        _plot_ws(ws, "WS par tenor (par currency bucket)")
    else:
        st.info("Pas de WS.")
    _step_box_end()

    # Step 3 — Intra-bucket K_b
    _step_box("③ Intra-bucket: K_b = sqrt(WS' ρ WS)", "ρ dépend de la règle (exp_absdiff ou basel_tenor) et du scénario.")
    if Kb:
        # tableau Kb par scénario
        rows = []
        for sc, d in Kb.items():
            for ccy, val in d.items():
                rows.append({"scenario": sc, "ccy": ccy, "K_b": float(val)})
        df_kb = pd.DataFrame(rows).sort_values(["scenario", "ccy"])
        df_kb["K_b"] = df_kb["K_b"].round(2)
        st.dataframe(df_kb, use_container_width=True, hide_index=True)

        # corr matrix viewer (optionnel, medium par défaut)
        sc_choice = st.selectbox("Voir la matrice de corrélation ρ (scénario)", ["low", "medium", "high"], index=1, key="sw_corr_scenario")
        ccy_choice = st.selectbox("Bucket (currency)", sorted(ws.keys()) if ws else ["—"], key="sw_corr_ccy")

        if ws and ccy_choice in ws:
            ten_nz = [float(T) for T in tenors if float(ws[ccy_choice].get(float(T), 0.0)) != 0.0]
            if len(ten_nz) >= 2:
                rho = corr_matrix_for_tenors(ten_nz, cfg_sw, scenario=sc_choice)
                df_rho = pd.DataFrame(rho, index=[f"{t}Y" for t in ten_nz], columns=[f"{t}Y" for t in ten_nz])
                with st.expander("Matrice ρ (tenors non-nuls)", expanded=False):
                    st.dataframe(df_rho, use_container_width=True)
            else:
                st.info("Pas assez de tenors non-nuls pour afficher ρ (il en faut ≥ 2).")
    else:
        st.info("Pas de Kb.")
    _step_box_end()

    # Step 4 — Inter-bucket totals
    _step_box("④ Inter-bucket: agrégation entre currencies", "K_total(scenario) = inter_bucket(Kb, S, gamma_scenario). K_final = max scénarios.")
    if totals:
        dfT = pd.DataFrame([{"scenario": k, "K_total": float(v)} for k, v in totals.items()]).sort_values("K_total", ascending=False)
        dfT["K_total"] = dfT["K_total"].round(2)
        st.dataframe(dfT, use_container_width=True, hide_index=True)
    else:
        st.info("Pas de totals.")
    _step_box_end()

# =========================================================
# GIRR BONDS
# =========================================================
with tab_bo:
    bo = res.get("girr_bonds", {}) or {}
    totals = bo.get("totals", {}) or {}
    worst = _worst_scenario(totals)

    cA, cB = st.columns([1, 2])
    with cA:
        st.metric("K_final", f"{float(bo.get('K_final', 0.0)):,.0f}".replace(",", " "))
        st.markdown(f'Worst scenario <span class="badge">{worst}</span>', unsafe_allow_html=True)
    with cB:
        if totals:
            st.write("Totaux par scénario:")
            st.dataframe(pd.DataFrame([totals]).rename_axis("scenario", axis=1), use_container_width=True)

    st.divider()

    ws = bo.get("WS", {}) or {}
    Kb = bo.get("Kb", {}) or {}
    cfg_bo = st.session_state["girr_cfg_bonds"]
    tenors = sorted(cfg_bo.rw_by_tenor.keys())

    _step_box("⓪ Bump & Reprice (sensibilités)", "PV0 et bump (détail trade-level dans les logs).")
    if show_logs_snippets and logs:
        snippet = _extract_between(logs, "GIRR — Bonds — Delta (SBM)", "RÉSUMÉ (par risk class)")
        if snippet.strip():
            st.code(snippet.strip(), language="text")
        else:
            st.info("Aucun extrait trouvé.")
    else:
        st.info("Activez 'Verbose' et relancez pour voir le détail bump&reprice.")
    _step_box_end()

    _step_box("① s_k (reconstruites)", "On reconstruit s_k ≈ WS_k / RW_k pour visualiser.")
    if ws:
        sens = _compute_girr_sens_from_ws(ws, cfg_bo, tenors)
        rows = []
        for ccy, d in sens.items():
            for T in tenors:
                rows.append({"ccy": ccy, "tenor": float(T), "s_k": float(d.get(float(T), 0.0))})
        df_s = pd.DataFrame(rows).sort_values(["ccy", "tenor"])
        df_s["s_k"] = df_s["s_k"].round(2)
        st.dataframe(df_s, use_container_width=True, hide_index=True)
    else:
        st.info("Pas de WS.")
    _step_box_end()

    _step_box("② Weighted sensitivities WS_k", "WS_k = RW_k * s_k.")
    if ws:
        _plot_ws(ws, "WS par tenor (par currency bucket)")
    else:
        st.info("Pas de WS.")
    _step_box_end()

    _step_box("③ Intra-bucket: K_b = sqrt(WS' ρ WS)", "Affiche K_b par scénario + matrice ρ optionnelle.")
    if Kb:
        rows = []
        for sc, d in Kb.items():
            for ccy, val in d.items():
                rows.append({"scenario": sc, "ccy": ccy, "K_b": float(val)})
        df_kb = pd.DataFrame(rows).sort_values(["scenario", "ccy"])
        df_kb["K_b"] = df_kb["K_b"].round(2)
        st.dataframe(df_kb, use_container_width=True, hide_index=True)

        sc_choice = st.selectbox("Voir la matrice de corrélation ρ (scénario)", ["low", "medium", "high"], index=1, key="bo_corr_scenario")
        ccy_choice = st.selectbox("Bucket (currency)", sorted(ws.keys()) if ws else ["—"], key="bo_corr_ccy")

        if ws and ccy_choice in ws:
            ten_nz = [float(T) for T in tenors if float(ws[ccy_choice].get(float(T), 0.0)) != 0.0]
            if len(ten_nz) >= 2:
                rho = corr_matrix_for_tenors(ten_nz, cfg_bo, scenario=sc_choice)
                df_rho = pd.DataFrame(rho, index=[f"{t}Y" for t in ten_nz], columns=[f"{t}Y" for t in ten_nz])
                with st.expander("Matrice ρ (tenors non-nuls)", expanded=False):
                    st.dataframe(df_rho, use_container_width=True)
            else:
                st.info("Pas assez de tenors non-nuls pour afficher ρ (il en faut ≥ 2).")
    else:
        st.info("Pas de Kb.")
    _step_box_end()

    _step_box("④ Inter-bucket: agrégation entre currencies", "K_final = max scénarios.")
    if totals:
        dfT = pd.DataFrame([{"scenario": k, "K_total": float(v)} for k, v in totals.items()]).sort_values("K_total", ascending=False)
        dfT["K_total"] = dfT["K_total"].round(2)
        st.dataframe(dfT, use_container_width=True, hide_index=True)
    else:
        st.info("Pas de totals.")
    _step_box_end()

# =========================================================
# RAW LOGS
# =========================================================
with tab_logs:
    if logs:
        st.code(logs, language="text")
    else:
        st.info("Pas de logs (verbose=False).")
