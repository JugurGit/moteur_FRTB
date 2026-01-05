from __future__ import annotations

import streamlit as st

from ui_common import apply_page_config, apply_pro_css, init_session_state

apply_page_config()
apply_pro_css()
init_session_state()

st.title("⚙️ Configs")
st.caption("Paramètres GIRR uniquement. Édition simple, stockée en session.")

sw_cfg = st.session_state["girr_cfg_swaps"]
bo_cfg = st.session_state["girr_cfg_bonds"]

# -------------------------------------------------------------------
# Valeurs FIXÉES (non éditables) demandées
# -------------------------------------------------------------------
# Choix par défaut: on fige à la valeur actuelle des configs en session
# (comme ça tu gardes tes valeurs 'demo' ou celles restaurées depuis Historique)
FIXED_SW_BUMP_BP = float(sw_cfg.bump_bp)
FIXED_SW_GAMMA_MED = float(sw_cfg.gamma_inter_ccy_med)

FIXED_BO_BUMP_BP = float(bo_cfg.bump_bp)
FIXED_BO_GAMMA_MED = float(bo_cfg.gamma_inter_ccy_med)

tab_sw, tab_bo = st.tabs(["GIRR Swaps", "GIRR Bonds"])

# -----------------------------
# TAB — GIRR Swaps
# -----------------------------
with tab_sw:
    st.subheader("GirrConfig — Swaps")

    # RW tenors editor
    rw_rows = [{"tenor": float(T), "rw": float(v)} for T, v in sorted(sw_cfg.rw_by_tenor.items())]
    rw_edit = st.data_editor(
        rw_rows,
        num_rows="dynamic",
        use_container_width=True,
        key="sw_rw_editor",
        column_config={
            "tenor": st.column_config.NumberColumn("tenor", format="%.4f"),
            "rw": st.column_config.NumberColumn("rw", format="%.8f"),
        },
    )

    c1, c2 = st.columns(2)
    with c1:
        specified = st.checkbox(
            "specified_currency_reduction",
            value=bool(sw_cfg.specified_currency_reduction),
            key="sw_specified_currency_reduction",
        )
    with c2:
        st.markdown("**Paramètres fixés**")
        st.text(f"bump_bp = {FIXED_SW_BUMP_BP:.8f}")
        st.text(f"gamma_inter_ccy_med = {FIXED_SW_GAMMA_MED:.6f}")
        st.text(f"scenario_rule = {sw_cfg.scenario_rule}")
        st.text(f"rho_rule = {sw_cfg.rho_rule}")
        st.text(f"rho_param = {float(sw_cfg.rho_param):.6f}")

    if st.button("Appliquer GIRR Swaps config", use_container_width=True, key="sw_apply_btn"):
        try:
            from girr import GirrConfig

            rw_by_tenor = {float(r["tenor"]): float(r["rw"]) for r in rw_edit}

            # On conserve scenario_rule / rho_rule / rho_param (mais non éditables)
            st.session_state["girr_cfg_swaps"] = GirrConfig(
                rw_by_tenor=rw_by_tenor,
                specified_currency_reduction=bool(specified),
                bump_bp=float(FIXED_SW_BUMP_BP),
                gamma_inter_ccy_med=float(FIXED_SW_GAMMA_MED),
                scenario_rule=str(sw_cfg.scenario_rule),
                rho_rule=str(sw_cfg.rho_rule),
                rho_param=float(sw_cfg.rho_param),
            )
            st.success("Config swaps mise à jour ✅")
        except Exception as e:
            st.error(f"Erreur config swaps: {e}")

# -----------------------------
# TAB — GIRR Bonds
# -----------------------------
with tab_bo:
    st.subheader("GirrConfig — Bonds")

    rw_rows = [{"tenor": float(T), "rw": float(v)} for T, v in sorted(bo_cfg.rw_by_tenor.items())]
    rw_edit = st.data_editor(
        rw_rows,
        num_rows="dynamic",
        use_container_width=True,
        key="bo_rw_editor",
        column_config={
            "tenor": st.column_config.NumberColumn("tenor", format="%.4f"),
            "rw": st.column_config.NumberColumn("rw", format="%.8f"),
        },
    )

    c1, c2 = st.columns(2)
    with c1:
        specified = st.checkbox(
            "specified_currency_reduction",
            value=bool(bo_cfg.specified_currency_reduction),
            key="bo_specified_currency_reduction",
        )
    with c2:
        st.markdown("**Paramètres fixés**")
        st.text(f"bump_bp = {FIXED_BO_BUMP_BP:.8f}")
        st.text(f"gamma_inter_ccy_med = {FIXED_BO_GAMMA_MED:.6f}")
        st.text(f"scenario_rule = {bo_cfg.scenario_rule}")
        st.text(f"rho_rule = {bo_cfg.rho_rule}")
        st.text(f"rho_param = {float(bo_cfg.rho_param):.6f}")

    if st.button("Appliquer GIRR Bonds config", use_container_width=True, key="bo_apply_btn"):
        try:
            from girr import GirrConfig

            rw_by_tenor = {float(r["tenor"]): float(r["rw"]) for r in rw_edit}

            st.session_state["girr_cfg_bonds"] = GirrConfig(
                rw_by_tenor=rw_by_tenor,
                specified_currency_reduction=bool(specified),
                bump_bp=float(FIXED_BO_BUMP_BP),
                gamma_inter_ccy_med=float(FIXED_BO_GAMMA_MED),
                scenario_rule=str(bo_cfg.scenario_rule),
                rho_rule=str(bo_cfg.rho_rule),
                rho_param=float(bo_cfg.rho_param),
            )
            st.success("Config bonds mise à jour ✅")
        except Exception as e:
            st.error(f"Erreur config bonds: {e}")
