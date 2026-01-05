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
- `pip` (ou `pipx`, `conda` si tu préfères)
- (Optionnel) `git`
