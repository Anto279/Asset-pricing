import numpy as np
import pandas as pd
from scipy.stats import norm
from src.utils.parametres import Parametres
from src.instruments.option_asiatique import OptionAsiatique

class AsianPricingModel:
    
    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = num_simulations

    def _price_geometric_analytic_exact(self, option: OptionAsiatique) -> float:
        """
        Implémente la formule fermée exacte (Black-Scholes modifié) pour 
        l'Option Asiatique à Moyenne GÉOMÉTRIQUE Discrète.
        Utilisée comme valeur théorique de référence (Expectation) pour le Control Variate.
        """
        # Récupération des paramètres
        S0 = option.spot
        K = option.parameters.get_strike()
        r = option.parameters.get_risk_free_rate()
        sigma = option.parameters.get_volatility()
        T = option.parameters.get_maturity()
        N = option.observation_steps # Nombre de fixings
        option_type = option.option_type

        dt = T / N
        
        # --- 1. Calcul des Moments Exacts de ln(G_T) ---
        
        # Le temps moyen arithmétique pour le drift
        # Somme(t_i) / N = dt * N(N+1)/2 / N = dt * (N+1)/2
        time_avg = dt * (N + 1) / 2
        
        # Paramètre nu (Drift du log-price)
        nu = r - 0.5 * sigma**2
        
        # Espérance de ln(G_T)
        mu_G = np.log(S0) + nu * time_avg
        
        # Variance de ln(G_T)
        # Formule exacte de la double somme des covariances min(ti, tj) pour pas constants
        variance_factor = (N + 1) * (2 * N + 1) / (6 * N**2)
        sigma_G_sq = (sigma**2) * T * variance_factor
        
        vol_G = np.sqrt(sigma_G_sq)

        # --- 2. Formule de Black-Scholes Modifiée ---
        
        # Forward ajusté : E[G_T] sous risque neutre
        # Note : G_T est log-normale, donc E[G_T] = exp(mu_G + 0.5 * sigma_G_sq)
        F_G = np.exp(mu_G + 0.5 * sigma_G_sq)
        
        d1 = (np.log(F_G / K) + 0.5 * sigma_G_sq) / vol_G
        d2 = d1 - vol_G
        
        # Discount factor
        df = np.exp(-r * T)
        
        if option_type == "call":
            price = df * (F_G * norm.cdf(d1) - K * norm.cdf(d2))
        elif option_type == "put":
            price = df * (K * norm.cdf(-d2) - F_G * norm.cdf(-d1))
        else:
            return 0.0
            
        return price

    def price_option_mc_control_variate(self, option: OptionAsiatique):
        """
        Pricing de l'Option Asiatique (Moyenne ARITHMÉTIQUE) 
        via Monte Carlo avec Variable de Contrôle (Moyenne Géométrique).
        """
        # Paramètres
        S0 = option.spot
        K = option.parameters.get_strike()
        r = option.parameters.get_risk_free_rate()
        sigma = option.parameters.get_volatility()
        T = option.parameters.get_maturity()
        N = option.observation_steps
        
        dt = T / N
        M = self.num_simulations
        
        # --- 1. Génération vectorisée des trajectoires (Paths) ---
        # Matrice (M simulations x N pas de temps)
        # On utilise cumsum sur les log-returns pour la rapidité
        
        Z = np.random.standard_normal((M, N))
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        log_returns = drift + diffusion
        log_paths = np.cumsum(log_returns, axis=1)
        
        # Reconstruction des prix S_t = S0 * exp(somme des log_returns)
        # paths est une matrice (M, N) où chaque ligne est une trajectoire de S_t1 à S_tN
        paths = S0 * np.exp(log_paths)
        
        # --- 2. Calcul des Moyennes Arithmétiques et Géométriques par chemin ---
        
        # Moyenne Arithmétique (cible)
        A_T = np.mean(paths, axis=1) # Vecteur de taille M
        
        # Moyenne Géométrique (variable de contrôle)
        # exp(mean(log(S))) est plus stable et rapide
        # On ajoute log(S0) car log_paths contient (ln(St) - ln(S0))
        # log_S_paths = np.log(paths) # ou log_paths + log(S0)
        G_T = np.exp(np.mean(np.log(paths), axis=1)) 
        
        # --- 3. Calcul des Payoffs (non actualisés) ---
        
        if option.option_type == "call":
            payoff_arith = np.maximum(A_T - K, 0)
            payoff_geo = np.maximum(G_T - K, 0)
        else:
            payoff_arith = np.maximum(K - A_T, 0)
            payoff_geo = np.maximum(K - G_T, 0)
            
        # --- 4. Variable de Contrôle : Optimisation du Beta ---
        
        # On cherche à estimer : E[Payoff_Arith]
        # Estimateur CV = Payoff_Arith - Beta * (Payoff_Geo - E[Payoff_Geo_Exact])
        
        # Calcul de la valeur théorique exacte du géométrique
        exact_geo_price = self._price_geometric_analytic_exact(option)
        # On désactualise le prix exact pour le comparer aux payoffs (en valeur future T)
        exact_geo_payoff_expectation = exact_geo_price * np.exp(r * T)
        
        # Matrice de covariance entre les payoffs Arithmétique et Géométrique
        cov_matrix = np.cov(payoff_arith, payoff_geo)
        var_geo = cov_matrix[1, 1]
        cov_ag = cov_matrix[0, 1]
        
        # Beta optimal (OLS) : Cov(A, G) / Var(G)
        if var_geo == 0:
            beta = 0 # Cas dégénéré (ex: volatilité nulle)
        else:
            beta = cov_ag / var_geo
            
        # --- 5. Estimateur Contrôlé ---
        
        # Y_control = Y_arith - beta * (Y_geo - E[Y_geo])
        # Note : On travaille sur les vecteurs
        control_variate_correction = beta * (payoff_geo - exact_geo_payoff_expectation)
        adjusted_payoffs = payoff_arith - control_variate_correction
        
        # Moyenne et Actualisation
        price_mc_cv = np.exp(-r * T) * np.mean(adjusted_payoffs)
        
        # --- Bonus : Métriques de précision ---
        # Calcul de la réduction de variance (pour info)
        var_arith = np.var(payoff_arith)
        var_adjusted = np.var(adjusted_payoffs)
        variance_reduction = 1 - (var_adjusted / var_arith) if var_arith > 0 else 0
        
        return price_mc_cv, variance_reduction
    
    # =========================================================
    # Greeks (MC bumping) — WITHOUT changing your pricer
    # =========================================================
    def greeks_mc_control_variate(
        self,
        option: OptionAsiatique,
        hS: float | None = None,
        hV: float = 0.01,         # +1 vol point (0.01)
        hR: float = 1e-4,         # +1 bp
        hT: float = 1/365.25,     # +1 day (in years)
        seed: int = 123
        ) -> dict:
        """
        Greeks via central finite differences using your existing
        price_option_mc_control_variate().

        Uses the SAME numpy RNG seed for each bumped pricing call -> Common Random Numbers.
        Conventions:
          Delta  = dV/dS
          Gamma  = d²V/dS²
          Vega   = dV/dsigma   (sigma in decimals)
          Rho    = dV/dr
          Theta  = - dV/dT     (time decay)
        """

        S0 = float(option.spot)
        K = float(option.parameters.get_strike())
        r = float(option.parameters.get_risk_free_rate())
        sigma = float(option.parameters.get_volatility())
        T = float(option.parameters.get_maturity())
        N = int(option.observation_steps)
        opt_type = str(option.option_type).lower()

        if hS is None:
            hS = max(1e-4, 0.01 * S0)  # 1% spot

        def _make_opt(S=None, T_=None, r_=None, sig_=None) -> OptionAsiatique:
            params = Parametres(
                maturity=float(T_ if T_ is not None else T),
                risk_free_rate=float(r_ if r_ is not None else r),
                strike=float(K),
                volatility=float(sig_ if sig_ is not None else sigma),
            )
            return OptionAsiatique(
                parameters=params,
                spot=float(S if S is not None else S0),
                option_type=opt_type,
                observation_steps=N
            )

        def _price(opt: OptionAsiatique) -> float:
            # Common Random Numbers without modifying your pricer:
            state = np.random.get_state()
            np.random.seed(seed)
            try:
                val, _ = self.price_option_mc_control_variate(opt)
                return float(val)
            finally:
                np.random.set_state(state)

        # Base
        V0 = _price(_make_opt())

        # Delta / Gamma (central)
        V_Sp = _price(_make_opt(S=S0 + hS))
        V_Sm = _price(_make_opt(S=max(1e-8, S0 - hS)))
        delta = (V_Sp - V_Sm) / (2.0 * hS)
        gamma = (V_Sp - 2.0 * V0 + V_Sm) / (hS ** 2)

        # Vega (central in sigma)
        V_vp = _price(_make_opt(sig_=sigma + hV))
        V_vm = _price(_make_opt(sig_=max(1e-8, sigma - hV)))
        vega = (V_vp - V_vm) / (2.0 * hV)         # dV/dsigma (sigma in decimals)
        vega_1pp = vega * 0.01                    # approx per +1 vol point

        # Rho (central in r)
        V_rp = _price(_make_opt(r_=r + hR))
        V_rm = _price(_make_opt(r_=r - hR))
        rho = (V_rp - V_rm) / (2.0 * hR)          # dV/dr
        rho_1bp = rho * 1e-4                      # approx per +1 bp

        # Theta (standard) = - dV/dT
        T_p = max(1e-6, T + hT)
        T_m = max(1e-6, T - hT)
        V_Tp = _price(_make_opt(T_=T_p))
        V_Tm = _price(_make_opt(T_=T_m))
        theta = - (V_Tp - V_Tm) / (2.0 * hT)       # per year
        theta_1day = theta / 365.25                # per day

        return {
            "Price": V0,
            "Delta": float(delta),
            "Gamma": float(gamma),
            "Vega": float(vega),
            "Vega_1pp": float(vega_1pp),
            "Theta": float(theta),
            "Theta_1day": float(theta_1day),
            "Rho": float(rho),
            "Rho_1bp": float(rho_1bp),
        }

    
    def get_paths_for_visualization(self, option: OptionAsiatique, num_paths_to_plot: int = 20):
        """
        Génère un échantillon de trajectoires pour la visualisation pédagogique.
        Retourne un DataFrame prêt pour Streamlit.
        """
        S0 = option.spot
        r = option.parameters.get_risk_free_rate()
        sigma = option.parameters.get_volatility()
        T = option.parameters.get_maturity()
        N = option.observation_steps
        dt = T / N
        
        # On simule seulement quelques chemins pour le graph
        Z = np.random.standard_normal((num_paths_to_plot, N))
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        log_paths = np.cumsum(log_returns, axis=1)
        paths = S0 * np.exp(log_paths)
        
        # Ajout du point de départ S0 à t=0
        paths = np.hstack([np.full((num_paths_to_plot, 1), S0), paths])
        
        # Création de l'axe temps
        time_axis = np.linspace(0, T, N + 1)
        
        # Formatage en DataFrame : Index = Temps, Colonnes = Chemin_1, Chemin_2...
        df_paths = pd.DataFrame(paths.T, index=time_axis, columns=[f"Path_{i+1}" for i in range(num_paths_to_plot)])
        
        return df_paths