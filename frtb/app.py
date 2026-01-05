# app.py
from __future__ import annotations

import streamlit as st

from ui_common import apply_page_config, apply_pro_css, init_session_state, summary_kpis

apply_page_config()
apply_pro_css()
init_session_state()

st.title("FRTB SA / SBM — Mini Dashboard")
st.caption("Interface Streamlit pour piloter le portfolio, le marché, les configs et lancer le moteur de calcul.")

col1, col2, col3, col4 = st.columns(4)
k = summary_kpis()
col1.metric("Equity trades", k["n_eq"])
col2.metric("GIRR swaps", k["n_sw"])
col3.metric("GIRR bonds", k["n_bo"])
col4.metric("Dernier run", k["last_run_status"])

st.divider()
st.markdown(
    """
### Navigation
Utilise les pages à gauche :
- **Overview** : résumé + état courant
- **Portfolio** : upload / édition interactive
- **Market** : courbes + FX
- **Configs** : paramètres Equity / GIRR
- **Run & Results** : exécution + résultats + logs
- **Export** : téléchargement des outputs

> Astuce : le moteur original `FRTBEngine.run()` imprime beaucoup — on capture ces logs et on les affiche.
"""
)

if st.session_state.get("last_logs"):
    with st.expander("Afficher les logs du dernier run", expanded=False):
        st.code(st.session_state["last_logs"], language="text")
