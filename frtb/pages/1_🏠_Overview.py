from __future__ import annotations

import streamlit as st
from ui_common import apply_page_config, apply_pro_css, init_session_state, summary_kpis

apply_page_config()
apply_pro_css()
init_session_state()

st.title("🏠 Overview")
st.markdown(
    """
Cette app expose un moteur **FRTB SA / SBM** via une interface :
- Portfolio (édition)
- Marché (courbes + FX)
- Configs (GIRR)
- Run + résultats + logs
- Export
- Historique

"""
)

k = summary_kpis()
c1, c2, c3 = st.columns(3)
c1.metric("Equity trades", k["n_eq"])
c2.metric("GIRR swaps", k["n_sw"])
c3.metric("GIRR bonds", k["n_bo"])

st.info("Allez dans **Portfolio** pour éditer, puis **Run & Results** pour exécuter.")
