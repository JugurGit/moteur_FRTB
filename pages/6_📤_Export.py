from __future__ import annotations

import streamlit as st

from ui_common import (
    apply_page_config,
    apply_pro_css,
    init_session_state,
    portfolio_to_csv,
    results_to_json,
)

apply_page_config()
apply_pro_css()
init_session_state()

st.title("📤 Export")
st.caption("Téléchargement du portfolio + résultats (JSON) + logs.")

p = st.session_state["portfolio"]
res = st.session_state.get("last_run")
logs = st.session_state.get("last_logs", "")

c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Portfolio (CSV)",
        data=portfolio_to_csv(p),
        file_name="portfolio.csv",
        mime="text/csv",
        use_container_width=True,
    )

with c2:
    if res is None:
        st.download_button(
            "⬇️ Résultats (JSON)",
            data="{}",
            file_name="results.json",
            mime="application/json",
            use_container_width=True,
            disabled=True,
        )
    else:
        st.download_button(
            "⬇️ Résultats (JSON)",
            data=results_to_json(res),
            file_name="results.json",
            mime="application/json",
            use_container_width=True,
        )

st.divider()

st.download_button(
    "⬇️ Logs (txt)",
    data=logs or "",
    file_name="run_logs.txt",
    mime="text/plain",
    use_container_width=True,
)
