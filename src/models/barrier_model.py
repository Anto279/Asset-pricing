import numpy as np
from scipy.stats import norm
from src.instruments.barrier_option import BarrierOption

class BarrierPricingModel:
    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = num_simulations

    # =================================================================
    # PARTIE 1 : MONTE CARLO AVEC PONT BROWNIEN
    # =================================================================
    def price_barrier_option_mc(self, option: BarrierOption, steps: int = 252, Z=None, U=None, seed: int | None = None):
        """
        Pricing via Monte Carlo avec correction de continuité (Brownian Bridge).
        Z/U/seed sont optionnels : utiles pour les grecques (common random numbers).
        """
        # 1. Extraction des paramètres
        S0 = option.spot
        K = option.parameters.get_strike()
        r = option.parameters.get_risk_free_rate()
        sigma = option.parameters.get_volatility()
        T = option.parameters.get_maturity()
        B = option.barrier
        
        is_up = "up" in option.barrier_type
        is_out = "out" in option.barrier_type
        
        if (is_up and S0 > B) or (not is_up and S0 < B):
            if is_out:
                return 0.0

        dt = T / steps
        M = self.num_simulations

        # --- AJOUT : RNG + common random numbers ---
        if seed is not None:
            rng = np.random.default_rng(seed)
            if Z is None:
                Z = rng.standard_normal((M, steps))
            if U is None:
                U = rng.uniform(0, 1, (M, steps))
        else:
            if Z is None:
                Z = np.random.standard_normal((M, steps))
            if U is None:
                U = np.random.uniform(0, 1, (M, steps))
        # -------------------------------------------

        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z

        log_paths = np.zeros((M, steps + 1))
        log_paths[:, 0] = np.log(S0)
        log_paths[:, 1:] = np.log(S0) + np.cumsum(drift + diffusion, axis=1)
        paths = np.exp(log_paths)

        if is_out:
            active_status = np.ones(M, dtype=bool)
        else:
            active_status = np.zeros(M, dtype=bool)

        # (SUPPRIMER l'ancien U = np.random.uniform(...), maintenant U est déjà défini)

        for t in range(steps):
            S_curr = paths[:, t]
            S_next = paths[:, t+1]

            if is_up:
                hit_discrete = (S_next >= B)
            else:
                hit_discrete = (S_next <= B)

            proba_hit = np.zeros(M)
            candidates = ~hit_discrete

            if np.any(candidates):
                log_s = np.log(S_curr[candidates] / B)
                log_s_next = np.log(S_next[candidates] / B)
                arg = -2 * log_s * log_s_next / (sigma**2 * dt)
                proba_hit[candidates] = np.exp(arg)

            hit_continuous = (U[:, t] < proba_hit) & candidates
            hit_step = hit_discrete | hit_continuous

            if is_out:
                active_status = active_status & (~hit_step)
            else:
                active_status = active_status | hit_step

        ST = paths[:, -1]
        if option.option_type == "call":
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        final_payoffs = payoffs * active_status.astype(float)
        price = np.exp(-r * T) * np.mean(final_payoffs)
        return price
    
    def compute_greeks_fd(
        self,
        option: BarrierOption,
        method: str = "mc",          # "mc" ou "analytic"
        steps: int = 252,
        spot_bump_pct: float = 0.01, # 1% du spot
        vol_bump: float = 0.01,      # +1% en vol (0.01 en absolu)
        rate_bump: float = 1e-4,     # +1bp = 0.0001
        time_bump: float = 1/365.25, # 1 jour en année
        seed: int = 123              # pour common random numbers en MC
    ):
        """
        Grecques par différences finies (robuste pour barrières).
        Retourne Delta, Gamma, Vega, Theta, Rho.
        - Vega est en "€/1.0 de vol" (si vol_bump=0.01 => ~€/1% vol)
        - Rho est en "€/1.0 de taux" (si rate_bump=1e-4 => ~€/1bp)
        """

        # --- helpers setters (compat Parametres) ---
        def _get_T():
            return option.parameters.get_maturity()

        def _set_maturity(T_new: float):
            if hasattr(option.parameters, "maturity"):
                option.parameters.maturity = T_new
            elif hasattr(option.parameters, "T"):
                option.parameters.T = T_new
            else:
                option.parameters.maturity = T_new  # fallback

        def _get_sigma():
            return option.parameters.get_volatility()

        def _set_volatility(sig_new: float):
            if hasattr(option.parameters, "volatility"):
                option.parameters.volatility = sig_new
            elif hasattr(option.parameters, "sigma"):
                option.parameters.sigma = sig_new
            else:
                option.parameters.volatility = sig_new

        def _get_r():
            return option.parameters.get_risk_free_rate()

        def _set_risk_free_rate(r_new: float):
            if hasattr(option.parameters, "risk_free_rate"):
                option.parameters.risk_free_rate = r_new
            elif hasattr(option.parameters, "r"):
                option.parameters.r = r_new
            else:
                option.parameters.risk_free_rate = r_new

        def _price_with_current_params(Z=None, U=None):
            if method == "analytic":
                return self.price_barrier_analytic(option)
            # method == "mc"
            return self.price_barrier_option_mc(option, steps=steps, Z=Z, U=U, seed=seed)

        # --- common random numbers for MC (reduce noise) ---
        Z = U = None
        if method == "mc":
            rng = np.random.default_rng(seed)
            Z = rng.standard_normal((self.num_simulations, steps))
            U = rng.uniform(0, 1, (self.num_simulations, steps))

        # Save originals
        S0 = option.spot
        sig0 = _get_sigma()
        r0 = _get_r()
        T0 = _get_T()

        # Base price
        V0 = _price_with_current_params(Z=Z, U=U)

        # -----------------
        # Delta / Gamma
        # -----------------
        hS = max(1e-6, spot_bump_pct * S0)
        try:
            option.spot = S0 + hS
            V_up = _price_with_current_params(Z=Z, U=U)

            option.spot = S0 - hS
            V_dn = _price_with_current_params(Z=Z, U=U)
        finally:
            option.spot = S0

        delta = (V_up - V_dn) / (2 * hS)
        gamma = (V_up - 2 * V0 + V_dn) / (hS ** 2)

        # -----------------
        # Vega
        # -----------------
        hV = max(1e-6, vol_bump)
        try:
            _set_volatility(sig0 + hV)
            V_sig_up = _price_with_current_params(Z=Z, U=U)

            _set_volatility(max(1e-8, sig0 - hV))
            V_sig_dn = _price_with_current_params(Z=Z, U=U)
        finally:
            _set_volatility(sig0)

        vega = (V_sig_up - V_sig_dn) / (2 * hV)

        # -----------------
        # Rho
        # -----------------
        hR = max(1e-8, rate_bump)
        try:
            _set_risk_free_rate(r0 + hR)
            V_r_up = _price_with_current_params(Z=Z, U=U)

            _set_risk_free_rate(r0 - hR)
            V_r_dn = _price_with_current_params(Z=Z, U=U)
        finally:
            _set_risk_free_rate(r0)

        rho = (V_r_up - V_r_dn) / (2 * hR)

        # -----------------
        # Theta (time decay)
        # Theta ~ (V(T - dt) - V(T)) / dt
        # -----------------
        theta = None
        dtT = time_bump
        if T0 - dtT > 1e-8:
            try:
                _set_maturity(T0 - dtT)
                V_shorter = _price_with_current_params(Z=Z, U=U)
            finally:
                _set_maturity(T0)
            theta = (V_shorter - V0) / dtT

        return {
            "Price": V0,
            "Delta": float(delta),
            "Gamma": float(gamma),
            "Vega": float(vega),
            "Theta": None if theta is None else float(theta),
            "Rho": float(rho),
        }



    # =================================================================
    # PARTIE 2 : FORMULE ANALYTIQUE (REINER-RUBINSTEIN)
    # =================================================================
    def price_barrier_analytic(self, option: BarrierOption):
        """
        Implémente les formules fermées de Reiner-Rubinstein (1991).
        Utilise les variables phi (Call/Put) et eta (Down/Up).
        """
        S = option.spot
        K = option.parameters.get_strike()
        r = option.parameters.get_risk_free_rate()
        sigma = option.parameters.get_volatility()
        T = option.parameters.get_maturity()
        B = option.barrier # H
        
        # 1. Configuration des variables binaires
        # phi : 1 pour Call, -1 pour Put
        phi = 1 if option.option_type == "call" else -1
        
        # eta : 1 pour Down, -1 pour Up (Attention à la convention)
        # Standard : Down means S > H (on descend vers H). Up means S < H.
        eta = 1 if "down" in option.barrier_type else -1
        
        # 2. Paramètres intermédiaires
        mu = (r - 0.5 * sigma**2) / sigma**2
        lam = np.sqrt(mu**2 + 2 * r / sigma**2) # Souvent mu + 1 si r constant
        
        # Facteurs d'actualisation et volatilité
        df = np.exp(-r * T)
        vol_sqrt_T = sigma * np.sqrt(T)
        
        # 3. Calcul des distances standardisées (x1, x2) et reflétées (y1, y2)
        # x1, x2 : Comme dans Black-Scholes standard
        x1 = (np.log(S / K) / vol_sqrt_T) + (1 + mu) * vol_sqrt_T
        x2 = x1 - vol_sqrt_T
        
        # y1, y2 : Reflétés par rapport à la barrière
        # Terme log(H^2 / (S*K))
        y1 = (np.log(B**2 / (S * K)) / vol_sqrt_T) + (1 + mu) * vol_sqrt_T
        y2 = y1 - vol_sqrt_T
        
        # 4. Calcul des "Briques" de prix
        
        # A. Prix Vanille Standard (V_vanilla)
        # Formule : phi * S * N(phi*x1) - phi * K * df * N(phi*x2)
        V_vanilla = phi * S * norm.cdf(phi * x1) - phi * K * df * norm.cdf(phi * x2)
        
        # B. Prix Reflété (V_reflet) (Rebate ou Image)
        # Formule : A * [phi * S * (H/S)^(2(mu+1)) * N(eta*y1) ... ]
        # Attention, ici eta joue un rôle dans la direction de l'intégration N()
        # Dans la formule standard RR pour les cas réguliers (K du bon côté) :
        # Down (eta=1) -> N(y). Up (eta=-1) -> N(-y). Donc N(eta * y).
        
        term_S = (B / S)**(2 * (mu + 1))
        term_K = (B / S)**(2 * mu)
        
        # Note : Pour les cas standards, on utilise eta.
        # Pour le reflet, on applique eta à y1 et y2
        V_reflet = phi * S * term_S * norm.cdf(eta * y1) - phi * K * df * term_K * norm.cdf(eta * y2)
        
        # 5. Assemblage final selon le type (In ou Out)
        
        # Cas "OUT" (Knock-Out)
        if "out" in option.barrier_type:
            # Prix = Vanille - Reflet
            # Cela fonctionne pour les cas standards (K > B pour Call Down, etc.)
            return max(0, V_vanilla - V_reflet)
            
        # Cas "IN" (Knock-In)
        elif "in" in option.barrier_type:
            # Prix = Reflet
            return max(0, V_reflet)
            
        return 0.0