import numpy as np
import pandas as pd
from datetime import datetime

# Imports nécessaires pour la construction des objets internes
from src.instruments.option_vanille import OptionVanille
from src.utils.parametres import Parametres
from src.models.black_scholes import BlackScholesModel
# On importe le loader uniquement pour le type hinting si besoin, 
# mais on évite l'import circulaire au runtime si possible.

class ImpliedVolatility:

    def __init__(self, market_price, option: 'OptionVanille', model: 'BlackScholesModel'):
        self.market_price = market_price
        self.option = option
        self.model = model

    def newton_raphson(self, initial_guess=0.2, tol=1e-5, max_iter=100):
        """Calcule la volatilité implicite via la méthode de Newton-Raphson."""
        implied_vol = initial_guess
        original_vol = self.option.parameters.get_volatility()
        
        for i in range(max_iter):
            self.option.parameters.volatility = implied_vol
            f_vol = self.model.price_option(self.option) - self.market_price
            vega = self.model.vega(self.option)

            if abs(f_vol) < tol:
                self.option.parameters.volatility = original_vol
                return implied_vol
            
            if abs(vega) < 1e-8:
                break

            implied_vol = implied_vol - f_vol / vega
        
        self.option.parameters.volatility = original_vol
        return None

    def dichotomie(self, lower_bound=1e-5, upper_bound=5.0, tol=1e-5, max_iter=200):
        """Calcule la volatilité implicite via la méthode de dichotomie.

        Version robuste :
        - Vérifie/élargit l'intervalle si besoin (bracketing)
        - S'arrête aussi si l'intervalle sur sigma devient très petit
        """
        original_vol = self.option.parameters.get_volatility()

        market = float(self.market_price)

        try:
            # Price at lower bound
            self.option.parameters.volatility = float(lower_bound)
            p_low = float(self.model.price_option(self.option))

            # Expand upper bound until price_high >= market (or cap)
            ub = float(upper_bound)
            self.option.parameters.volatility = ub
            p_high = float(self.model.price_option(self.option))

            cap = 10.0
            while p_high < market and ub < cap:
                ub *= 2.0
                self.option.parameters.volatility = ub
                p_high = float(self.model.price_option(self.option))

            # If still not bracketed, no solution under this model/inputs
            if market < p_low - tol or p_high + tol < market:
                self.option.parameters.volatility = original_vol
                return None

            lower_bound = float(lower_bound)
            upper_bound = float(ub)

            for _ in range(max_iter):
                mid_vol = (lower_bound + upper_bound) / 2.0
                self.option.parameters.volatility = mid_vol
                price_mid = float(self.model.price_option(self.option))

                if abs(price_mid - market) < tol:
                    self.option.parameters.volatility = original_vol
                    return mid_vol

                # Standard bisection update
                if price_mid < market:
                    lower_bound = mid_vol
                else:
                    upper_bound = mid_vol

                # If sigma interval is tiny, stop
                if (upper_bound - lower_bound) < 1e-10:
                    self.option.parameters.volatility = original_vol
                    return (lower_bound + upper_bound) / 2.0

        finally:
            # Always restore
            self.option.parameters.volatility = original_vol

        return None

    # On utilise une méthode statique car get_surface() car elle implique de créer plusieurs instances d'option 
    # et d'ImpliedVolatility, donc pas besoin d'une instance existante.

    @staticmethod
    def get_surface(ticker: str, loader, model: 'BlackScholesModel', max_strikes=20, progress_callback=None) -> pd.DataFrame:
        """
        Génère un DataFrame contenant les données de la surface de volatilité.
        
        Args:
            ticker: Le symbole de l'actif (ex: 'SPY').
            loader: Une instance de MarketDataLoader.
            model: Une instance de BlackScholesModel.
            max_strikes: Nombre maximum de strikes à traiter par maturité (pour la performance).
            progress_callback: Une fonction optionnelle pour mettre à jour une barre de progression (ex: st.progress).
        """
        
        # 1. Récupération des données initiales
        try:
            spot = loader.get_equity_spot(ticker)
            expirations = loader.get_option_expirations(ticker)
        except Exception as e:
            print(f"Erreur data init: {e}")
            return pd.DataFrame()

        if not expirations:
            return pd.DataFrame()

        surface_data = []
        
        # On prend toutes les maturités existantes sauf la première
        target_expirations = expirations[1:]  

        for idx, exp_date_str in enumerate(target_expirations):
            
            # Mise à jour de la barre de progression externe si fournie
            if progress_callback:
                progress_callback((idx + 1) / len(target_expirations))

            # Calcul du temps à maturité T
            try:
                exp_date = datetime.strptime(exp_date_str, "%Y-%m-%d")
                days_to_maturity = (exp_date - datetime.now()).days
                if days_to_maturity < 1: 
                    continue
                T = days_to_maturity / 365.0
            except ValueError:
                continue
            
            # Taux sans risque
            try:
                r = loader.get_risk_free_rate_from_curve(T)
            except:
                r = 0.04

            # Récupération de la chaîne (Calls uniquement pour simplifier)
            try:
                chain = loader.get_option_chain_data(ticker, exp_date_str)
                df_options = chain['calls']
            except:
                continue
                
            # Filtrage et échantillonnage
            df_options = df_options[df_options['volume'] > 0].copy()
            if len(df_options) > max_strikes:
                # On prend un échantillon régulièrement espacé
                step = int(len(df_options) / max_strikes)
                df_options = df_options.iloc[::step]

            for _, row in df_options.iterrows():
                strike = row['strike']
                # Prix LastPrice
                market_price = row['lastPrice']
                
                # Construction des objets pour le calcul unitaire
                # On met une vol arbitraire (0.2) qui sera écrasée par le solveur
                params = Parametres(strike=strike, maturity=T, risk_free_rate=r, volatility=0.2, dividend=0.0)
                opt = OptionVanille(parameters=params, spot=spot, option="call")
                
                # Instanciation pour ce point précis
                iv_calculator = ImpliedVolatility(market_price, opt, model)
                
                # Calcul
                iv = iv_calculator.newton_raphson()
                
                # Filtrage des valeurs aberrantes (volatilités < 1% ou > 200%)
                if iv is not None and 0.01 < iv < 2.0:
                    surface_data.append({
                        "Strike": strike,
                        "Maturity": T,
                        "Implied Volatility": iv,
                        "Moneyness": strike / spot
                    })

        return pd.DataFrame(surface_data)