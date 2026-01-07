# FRTB SA / SBM — Mini Dashboard (Streamlit)

Dashboard Streamlit multi-pages qui expose un **mini moteur FRTB SA / SBM** :
- **Equity** : Delta + Vega + Curvature (via Black-Scholes)
- **GIRR** : Delta (swaps + bonds) via **bump & reprice** sur zéro-rates
- **Run & Results** : exécution + logs capturés + vues step-by-step
- **Export** : portfolio/results/logs
- **Historique** : runs persistés en **SQLite** (snapshot + KPIs + outputs)

> ⚠️ Projet à but illustratif : paramètres, mappings et market “demo” ne sont pas une implémentation complète du texte réglementaire.

👉 Démo en ligne : **https://boudarene-moteurfrtb.streamlit.app/**
---

## 1) Prérequis

- **Python 3.10+** (recommandé)

### 2) Récupérer le projet
#### Option A — via Git
```bash
git clone <URL_DU_REPO>
cd <NOM_DU_REPO>
```

#### Option B — via ZIP
- Télécharger le ZIP depuis GitHub
- Le dézippez
- Ouvrir un terminal dans le dossier du projet

### 3) Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4) Lancer l’application Streamlit
```bash
streamlit run app.py
```
Streamlit va afficher une URL du type :
- Local: http://localhost:8501

### 5) Utilisation rapide
- Ouvrir Portfolio : éditer un portfolio.
- Ouvre Market : ajuster les courbes (zéro rates) et FX.
- Ouvre Configs : ajuster les risk weights GIRR (Swaps/Bonds).
- Ouvre Run & Results : lancer le moteur et consulter les steps + logs.
- Ouvre Historique : retrouver les runs (SQLite) + restaurer un snapshot.
- Ouvre Export : télécharger portfolio/results/logs.

### 6) Lancer le moteur en ligne de commande (sans Streamlit)

Le script main.py exécute un run “console” :

**Mode démo**
```bash
python main.py
```
