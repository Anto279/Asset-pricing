from src.utils.parametres import Parametres
from src.market_data.loader import MarketDataLoader
import numpy as np

class Future:
    def __init__(self, parameters: Parametres, spot: float):
        
        self.maturity = parameters.get_maturity()
        self.rate = parameters.get_risk_free_rate()
        self.spot = spot
        self.dividend = parameters.get_dividend()

        self.parameters = parameters
    
    def pricing_future(self):
        """
        Calcule le prix Forward théorique actuel (F0).
        F = S * exp((r - q) * T)
        """
        T = self.maturity
        r = self.rate
        St = self.spot
        q = self.dividend

        price = St * np.exp((r - q) * T) 
        return price

    def payoff(self, spot_at_maturity: float) -> float:
        """
        Calcule le payoff du Future à maturité pour le produit structuré.
        Payoff = S_T - K 
        (Si on a acheté le future au prix K, on gagne la différence).
        """
        # On récupère le strike (prix d'achat du future).
        # Si par hasard il n'est pas défini (cas rare), on prend 0.
        try:
            K = self.parameters.get_strike()
        except:
            K = 0.0
            
        return spot_at_maturity - K
    
    def delta(self) -> float:
        """
        Delta du Future = exp((r - q) * T)
        """
        T = self.maturity
        r = self.rate
        q = self.dividend
        return np.exp((r - q) * T)

    def gamma(self) -> float:
        """
        Gamma du Future = 0 (Linéaire)
        """
        return 0.0

    def vega(self) -> float:
        """
        Vega du Future = 0 (Pas de sensibilité à la volatilité)
        """
        return 0.0