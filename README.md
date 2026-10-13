# Pricing Project — ENSAI 

## Élèves : Antonin Meudic, Tom O'Connor

Application Streamlit de **pricing d’instruments financiers** et d’outils pédagogiques : obligations, swaps, futures, options (vanilles, asiatiques, barrières), calibration de volatilité implicite, surface de volatilité, et structuration de produits (legs).

---

## ✨ Fonctionnalités

### 1) Pricer (multi-instruments)
- **Obligation (Bond)** : pricing par actualisation des flux, avec possibilité de taux manuel ou **courbe FRED (US Treasuries)** + spread “profil de risque”.
- **Swap de taux** : calcul du taux de swap “au pair”, valuation jambe fixe / jambe flottante, sensibilités (DV01, convexity, carry) + affichage des taux utilisés.
- **Future / Forward sur actions** : \( F_0 = S_0 e^{(r-q)T} \) + historique des prix.
- **Option vanille sur actions**
  - **Européenne** :
    - Black–Scholes (analytique)
    - Monte Carlo (GBM)
    - Heston (Monte Carlo, variance stochastique)
  - **Américaine** :
    - Arbre binomial CRR (early exercise)
    - Monte Carlo Longstaff–Schwartz (LSM)
  - **Affichages** : prix, historique, trajectoires (GBM / Heston / LSM), payoff, et grecques (selon méthode).
- **Option asiatique** : Monte Carlo avec **control variate** + grecques par bumping.
- **Option barrière** :
  - Monte Carlo avec correction **Pont Brownien (Brownian Bridge)**
  - Formule fermée (type Reiner–Rubinstein / Merton selon l’implémentation)
  - Grecques par différences finies

### 2) Produits structurés
- Construction d’un produit comme somme de **legs** (options, obligation, sous-jacent proxy).
- Onglet **stratégies** (straddle, butterfly, spreads…) + import automatique des legs dans un builder.
- Onglet **builder** pour composer un produit sur mesure, pricer, et tracer payoff / PnL.

### 3) Calibration de volatilité implicite
- Récupère un prix de marché (Yahoo Finance) et calcule la **volatilité implicite** via :
  - Newton–Raphson
  - Dichotomie
- Visualisation “prix BS vs vol” + point solution.

### 4) Surface de volatilité implicite
- Génération d’une grille \( (K, T, \sigma_{impl}) \) et affichage 3D (Plotly).

---

## Comment exécuter ?

**Version en une phrase**
> Pour exécuter l’application, installe d’abord les dépendances avec `pip install -r requirements.txt`, puis lance `streamlit run main_app.py` dans le terminal.

Petite remarque : on peut aussi lancer avec `python -m streamlit run main_app.py`, c’est équivalent.