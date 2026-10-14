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
            
        # Scale for UI (more interpretable)
        vega_1pct = vega * 0.01          # €/ +1 vol point
        rho_1bp   = rho  * 1e-4          # €/ +1 bp
        rho_1pct  = rho  * 0.01          # €/ +1% rate

        # If you want "theta per day" (NOT annualized), don't divide:
        # here theta_raw is dV/dT (per year). Convert to per day:
        theta_per_day = None if theta is None else theta / 365.25

        return {
            "Price": V0,
            "Delta": float(delta),
            "Gamma": float(gamma),

            # --- UI-friendly aliases (units matter!) ---
            # Vega:   €/ +1% vol  ("1 vol point")
            # Rho:    €/ +1 bp    (0.01% rate)
            # Theta:  €/ day
            "Vega": float(vega_1pct),
            "Rho": float(rho_1bp),
            "Theta": None if theta_per_day is None else float(theta_per_day),

            # raw derivatives
            "Vega_raw": float(vega),
            "Rho_raw": float(rho),
            "Theta_raw": None if theta is None else float(theta),

            # scaled for display
            "Vega_1pct": float(vega_1pct),
            "Rho_1bp": float(rho_1bp),
            "Rho_1pct": float(rho_1pct),
            "Theta_per_day": None if theta_per_day is None else float(theta_per_day),
        }



    # =================================================================
    # PARTIE 2 : FORMULE ANALYTIQUE (REINER-RUBINSTEIN)
    # =================================================================
    def price_barrier_analytic(self, option: BarrierOption):
        """
        Implémente les formules fermées de Reiner-Rubinstein (1991),
        dans la présentation de Haug (The Complete Guide to Option Pricing Formulas).

        Le prix s'écrit comme une combinaison de 4 briques A, B, C, D :
          - A : option vanille (Black-Scholes)
          - B : vanille "coupée" au niveau de la barrière (terme en ln(S/H))
          - C : option "reflet" (principe de réflexion, terme en ln(H^2 / (S*K)))
          - D : reflet "coupé" au niveau de la barrière (terme en ln(H/S))
        La combinaison dépend du type (call/put), de la direction (down/up),
        du mécanisme (in/out) et de la position du strike par rapport à la barrière.
        Rebate = 0. Dividende continu q pris en compte s'il est renseigné.
        """
        S = float(option.spot)
        K = float(option.parameters.get_strike())
        r = float(option.parameters.get_risk_free_rate())
        sigma = float(option.parameters.get_volatility())
        T = float(option.parameters.get_maturity())
        H = float(option.barrier)
        q = option.parameters.get_dividend() if hasattr(option.parameters, "get_dividend") else 0.0
        b = r - q  # cost of carry

        is_call = option.option_type == "call"
        is_down = "down" in option.barrier_type
        is_out = "out" in option.barrier_type

        # Barrière déjà franchie au départ : l'option Out est morte, l'option In est une vanille
        already_hit = (is_down and S <= H) or ((not is_down) and S >= H)

        # 1. Paramètres intermédiaires
        mu = (b - 0.5 * sigma**2) / sigma**2
        vol_sqrt_T = sigma * np.sqrt(T)
        df_r = np.exp(-r * T)        # actualisation du strike
        df_q = np.exp((b - r) * T)   # = exp(-qT), appliqué au spot

        # 2. Distances standardisées
        x1 = np.log(S / K) / vol_sqrt_T + (1 + mu) * vol_sqrt_T
        x2 = np.log(S / H) / vol_sqrt_T + (1 + mu) * vol_sqrt_T
        y1 = np.log(H**2 / (S * K)) / vol_sqrt_T + (1 + mu) * vol_sqrt_T
        y2 = np.log(H / S) / vol_sqrt_T + (1 + mu) * vol_sqrt_T

        # 3. Les 4 briques (phi = +1 call / -1 put, eta = +1 down / -1 up)
        def A(phi):
            return phi * S * df_q * norm.cdf(phi * x1) - phi * K * df_r * norm.cdf(phi * x1 - phi * vol_sqrt_T)

        def B(phi):
            return phi * S * df_q * norm.cdf(phi * x2) - phi * K * df_r * norm.cdf(phi * x2 - phi * vol_sqrt_T)

        def C(phi, eta):
            return (phi * S * df_q * (H / S)**(2 * (mu + 1)) * norm.cdf(eta * y1)
                    - phi * K * df_r * (H / S)**(2 * mu) * norm.cdf(eta * y1 - eta * vol_sqrt_T))

        def D(phi, eta):
            return (phi * S * df_q * (H / S)**(2 * (mu + 1)) * norm.cdf(eta * y2)
                    - phi * K * df_r * (H / S)**(2 * mu) * norm.cdf(eta * y2 - eta * vol_sqrt_T))

        phi = 1 if is_call else -1
        vanilla = A(phi)

        if already_hit:
            return 0.0 if is_out else max(0.0, float(vanilla))

        # 4. Prix des options "In" (Haug, rebate nul)
        if is_call and is_down:
            price_in = C(1, 1) if K > H else A(1) - B(1) + D(1, 1)
        elif is_call and not is_down:
            price_in = A(1) if K > H else B(1) - C(1, -1) + D(1, -1)
        elif (not is_call) and is_down:
            price_in = B(-1) - C(-1, 1) + D(-1, 1) if K > H else A(-1)
        else:
            price_in = A(-1) - B(-1) + D(-1, -1) if K > H else C(-1, -1)

        # 5. Parité In + Out = Vanille
        price = vanilla - price_in if is_out else price_in

        # max(0, .) uniquement pour absorber les erreurs d'arrondi (~1e-15)
        return max(0.0, float(price))
