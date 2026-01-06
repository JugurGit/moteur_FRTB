# app.py
from __future__ import annotations

import streamlit as st

from ui_common import apply_page_config, apply_pro_css, init_session_state, summary_kpis

apply_page_config()
apply_pro_css()
init_session_state()

st.title("Moteur FRTB SA / SBM ")
st.caption("Interface Streamlit pour piloter le portfolio, le marché, les configs et lancer le moteur de calcul.")

st.markdown("### 🏦 Contexte - Aggrégation reporting FRTB")

st.info(
    """
Lors de mon stage de fin d’études chez **Banque Palatine** (équipe **Risque de Marché & Risque de Contrepartie**),
j’ai travaillé sur la **consolidation / agrégation de reportings FRTB** du département Risques Financiers.

Dans ce contexte, les calculs étaient réalisés via une **librairie Python** produisant des sorties structurées,
et l’enjeu côté reporting consistait à **standardiser les inputs/outputs**, **assembler** l’information et la
restituer sous un **format Excel consolidé**, exploitable pour le pilotage.
""",
    icon="📌",
)

st.warning(
    """
Je ne dispose pas des **données internes** ni de la **documentation** nécessaires
pour illustrer les traitements de manière “réelle”.  
Ce projet est donc une **réplique** : il ne reproduit pas l’environnement interne, mais il recrée
la **chaîne de production** et la logique de reporting.
""",
    icon="⚠️",
)

st.markdown("### 🎯 Ce que démontre ce mini-projet (workflow end-to-end)")

cA, cB, cC, cD = st.columns(4)
with cA:
    st.markdown("**1) Inputs normalisés**")
    st.caption("Portfolio • Market snapshot • Configs réglementaires")
with cB:
    st.markdown("**2) Moteur FRTB SA/SBM**")
    st.caption("Sensibilités • WS • Agrégations intra/inter-bucket")
with cC:
    st.markdown("**3) Restitution reporting-ready**")
    st.caption("Tables • Graphiques • Matrices ρ • Steps explicables")
with cD:
    st.markdown("**4) Traçabilité & rejouabilité**")
    st.caption("Logs capturés • Runs historisés • Snapshots restaurables")

st.success(
    """
✅ **En résumé** : les données sont **synthétiques** et le périmètre est **pédagogique** (Equity + GIRR),
mais l’application illustre concrètement ce que j’ai fait en stage :
**structurer** les entrées/sorties d’un moteur, **consolider** un reporting, et assurer la **reproductibilité**.
""",
    icon="✅",
)

with st.expander("🔎 Comment je m’y suis pris (approche “industrie du reporting”)", expanded=False):
    st.markdown(
        """
- **Contrat de données** : définition d’un format pivot pour le portfolio (CSV) et d’un snapshot marché (courbes/FX).  
- **Séparation calcul / restitution** : le moteur renvoie des résultats structurés ; l’UI se charge de la mise en forme.  
- **Explicabilité** : affichage étape-par-étape (WS, Kb, totaux par scénario), matrices de corrélation et graphiques.  
- **Audit trail** : capture des logs + historisation SQLite des runs (statut, KPIs, snapshots, exports) pour rejouer/comparer.
"""
    )

st.divider()



if st.session_state.get("last_logs"):
    with st.expander("Afficher les logs du dernier run", expanded=False):
        st.code(st.session_state["last_logs"], language="text")
