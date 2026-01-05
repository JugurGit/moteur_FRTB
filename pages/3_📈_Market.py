from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from ui_common import (
    apply_page_config,
    apply_pro_css,
    curve_to_rows,
    fx_to_rows,
    init_session_state,
    rows_to_curve,
    rows_to_fx,
    update_market_curves,
)

apply_page_config()
apply_pro_css()
init_session_state()

st.title("📈 Market")
st.caption("Édition des courbes (zéro rates) et FX. Le marché courant est stocké en session.")

mkt = st.session_state["market"]

tabs = st.tabs(["Curves", "FX", "Charts"])

with tabs[0]:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("EUR ZeroCurve")
        eur_rows = curve_to_rows(mkt.curves["EUR"])
        eur_edit = st.data_editor(eur_rows, num_rows="dynamic", use_container_width=True)
        st.session_state["_eur_curve_rows"] = eur_edit

    with c2:
        st.subheader("USD ZeroCurve")
        usd_rows = curve_to_rows(mkt.curves["USD"])
        usd_edit = st.data_editor(usd_rows, num_rows="dynamic", use_container_width=True)
        st.session_state["_usd_curve_rows"] = usd_edit

    if st.button("Appliquer les courbes", use_container_width=True):
        try:
            eur = rows_to_curve(st.session_state["_eur_curve_rows"])
            usd = rows_to_curve(st.session_state["_usd_curve_rows"])
            st.session_state["market"] = update_market_curves(mkt, {"EUR": eur, "USD": usd})
            st.success("Courbes mises à jour ✅")
        except Exception as e:
            st.error(f"Erreur courbe: {e}")

with tabs[1]:
    st.subheader("FX")
    fx_rows = fx_to_rows(mkt.fx)
    fx_edit = st.data_editor(
        fx_rows,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "from": st.column_config.TextColumn("from"),
            "to": st.column_config.TextColumn("to"),
            "rate": st.column_config.NumberColumn("rate"),
        },
    )
    if st.button("Appliquer FX", use_container_width=True):
        try:
            fx = rows_to_fx(fx_edit)
            st.session_state["market"] = update_market_curves(mkt, {}, fx=fx)
            st.success("FX mis à jour ✅")
        except Exception as e:
            st.error(f"Erreur FX: {e}")

with tabs[2]:
    st.subheader("Visualisation")
    c1, c2 = st.columns(2)

    # re-lire le market depuis session (au cas où il vient d’être modifié)
    mkt2 = st.session_state["market"]

    for ccy, col in [("EUR", c1), ("USD", c2)]:
        with col:
            curve = mkt2.curves[ccy]
            st.markdown(f"**{ccy}**")

            ten = np.array(curve.tenors, dtype=float)
            z = np.array(curve.zeros, dtype=float)

            df_zero = pd.DataFrame({"tenor": ten, "zero": z})
            st.line_chart(df_zero, x="tenor", y="zero", use_container_width=True)

            grid = np.linspace(float(ten[0]), float(ten[-1]), 80)
            dfs = np.array([curve.df(float(t)) for t in grid], dtype=float)

            df_df = pd.DataFrame({"t": grid, "df": dfs})
            st.line_chart(df_df, x="t", y="df", use_container_width=True)
