from src.utils.parametres import Parametres
import numpy as np
import pandas as pd

class OptionAsiatique:
    def __init__(self, parameters: Parametres, spot: float, option_type: str = "call", observation_steps: int = 252):
        """
        Args:
            parameters (Parametres): Contient strike, maturity, vol, r...
            spot (float): Prix spot actuel S0.
            option_type (str): "call" ou "put".
            observation_steps (int): Nombre de points de fixing pour la moyenne (ex: 252 pour une moyenne quotidienne sur 1 an).
        """
        self.spot = spot
        self.option_type = option_type.lower()
        self.parameters = parameters
        self.observation_steps = observation_steps

    def payoff_arithmetic(self, prices_path: np.array):
        """
        Calcule le payoff basé sur la moyenne arithmétique d'un chemin de prix.
        Args:
            prices_path (np.array): Tableau des prix simulés [S_t1, S_t2, ..., S_tN]
        """
        # Calcul de la moyenne arithmétique sur le chemin
        avg_price = np.mean(prices_path)
        K = self.parameters.get_strike()

        if self.option_type == "call":
            return max(avg_price - K, 0)
        elif self.option_type == "put":
            return max(K - avg_price, 0)
        return 0.0

    def payoff_geometric(self, prices_path: np.array):
        """
        Calcule le payoff basé sur la moyenne géométrique (pour le Control Variate).
        """
        # Moyenne géométrique = exp( moyenne( log(S) ) )
        # C'est numériquement plus stable que prod(S)^(1/N)
        log_prices = np.log(prices_path)
        avg_log = np.mean(log_prices)
        geo_avg = np.exp(avg_log)
        
        K = self.parameters.get_strike()

        if self.option_type == "call":
            return max(geo_avg - K, 0)
        elif self.option_type == "put":
            return max(K - geo_avg, 0)
        return 0.0