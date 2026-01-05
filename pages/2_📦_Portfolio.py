from __future__ import annotations

import streamlit as st

from ui_common import (
    apply_page_config,
    apply_pro_css,
    bonds_rows_from_portfolio,
    equity_rows_from_portfolio,
    init_session_state,
    portfolio_from_rows,
    portfolio_to_csv,
    swaps_rows_from_portfolio,
)
from demo import demo_portfolio

apply_page_config()
apply_pro_css()
init_session_state()

st.title("📦 Portfolio")
st.caption("Édition interactive du portfolio. Le portfolio courant est stocké en session.")

# Top actions
colA, colB = st.columns([1, 1])
with colA:
    if st.button("Reset → Demo portfolio", use_container_width=True, key="pf_reset_demo_top"):
        st.session_state["portfolio"] = demo_portfolio()
        st.success("Portfolio réinitialisé ✅")

with colB:
    st.download_button(
        "⬇️ Télécharger le portfolio courant (CSV)",
        data=portfolio_to_csv(st.session_state["portfolio"]),
        file_name="portfolio.csv",
        mime="text/csv",
        use_container_width=True,
        key="pf_download_csv",
    )

st.divider()

p = st.session_state["portfolio"]

tab1, tab2, tab3 = st.tabs(["Equity Calls", "GIRR Swaps", "GIRR Bonds"])

with tab1:
    rows = equity_rows_from_portfolio(p)
    edited = st.data_editor(
        rows,
        num_rows="dynamic",
        use_container_width=True,
        key="pf_eq_editor",
        column_config={
            "bucket": st.column_config.NumberColumn("bucket", step=1),
            "N": st.column_config.NumberColumn("N"),
            "S0": st.column_config.NumberColumn("S0"),
            "K": st.column_config.NumberColumn("K"),
            "T": st.column_config.NumberColumn("T"),
            "r": st.column_config.NumberColumn("r"),
            "q_repo": st.column_config.NumberColumn("q_repo"),
            "sigma": st.column_config.NumberColumn("sigma"),
        },
    )
    st.session_state["_eq_rows"] = edited

with tab2:
    rows = swaps_rows_from_portfolio(p)
    edited = st.data_editor(
        rows,
        num_rows="dynamic",
        use_container_width=True,
        key="pf_sw_editor",
        column_config={
            "ccy": st.column_config.TextColumn("ccy"),
            "notional": st.column_config.NumberColumn("notional"),
            "maturity": st.column_config.NumberColumn("maturity", step=1),
            "receive_fixed": st.column_config.CheckboxColumn("receive_fixed"),
        },
    )
    st.session_state["_sw_rows"] = edited

with tab3:
    rows = bonds_rows_from_portfolio(p)
    edited = st.data_editor(
        rows,
        num_rows="dynamic",
        use_container_width=True,
        key="pf_bo_editor",
        column_config={
            "ccy": st.column_config.TextColumn("ccy"),
            "notional": st.column_config.NumberColumn("notional"),
            "coupon_rate": st.column_config.NumberColumn("coupon_rate"),
            "maturity": st.column_config.NumberColumn("maturity", step=1),
        },
    )
    st.session_state["_bo_rows"] = edited

st.divider()

c1, c2 = st.columns([1, 2])
with c1:
    if st.button("✅ Appliquer les modifications", use_container_width=True, key="pf_apply_btn"):
        try:
            eq_rows = st.session_state.get("_eq_rows", [])
            sw_rows = st.session_state.get("_sw_rows", [])
            bo_rows = st.session_state.get("_bo_rows", [])
            st.session_state["portfolio"] = portfolio_from_rows(eq_rows, sw_rows, bo_rows)
            st.success("Portfolio mis à jour ✅")
        except Exception as e:
            st.error(f"Erreur de validation: {e}")

with c2:
    st.markdown(
        '<span class="small-note">Astuce : éditez 1–2 trades puis lancez un run pour valider la chaîne complète.</span>',
        unsafe_allow_html=True,
    )
