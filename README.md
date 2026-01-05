# FRTB SA / SBM — Mini Dashboard (Streamlit)

Dashboard Streamlit multi-pages qui expose un **mini moteur FRTB SA / SBM** :
- **Equity** : Delta + Vega + Curvature (via Black-Scholes)
- **GIRR** : Delta (swaps + bonds) via **bump & reprice** sur zéro-rates
- **Run & Results** : exécution + logs capturés + vues step-by-step
- **Export** : portfolio/results/logs
- **Historique** : runs persistés en **SQLite** (snapshot + KPIs + outputs)

> ⚠️ Projet pédagogique / démo : paramètres, mappings et market “demo” ne sont pas une implémentation complète du texte réglementaire.

---

## 1) Prérequis

- **Python 3.10+** (recommandé)

## 2) Structure

├── app.py
├── main.py
├── ui_common.py
├── history_db.py
├── curves.py
├── market.py
├── portfolio.py
├── equity.py
├── girr.py
├── engine.py
├── demo.py
├── export_projet.py
├── frtb_history.sqlite3            # généré automatiquement après un run (local)
├── pages/
│   ├── 1_🏠_Overview.py
│   ├── 2_📦_Portfolio.py
│   ├── 3_📈_Market.py
│   ├── 4_⚙️_Configs.py
│   ├── 5_🧮_Run_Results.py
│   ├── 6_📤_Export.py
│   ├── 7_🕘_Historique.py
│   ├── 8_🧾_Documentation.py
│   ├── code_docs.py
│   └── docs_registry.json
└── .streamlit/
    └── config.toml
