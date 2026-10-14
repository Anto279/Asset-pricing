from src.utils.parametres import Parametres

class Bonds:
    
    def __init__(self, parameters: Parametres):
        
        # On utilise des 'try/except' ou des valeurs par défaut 
        # au cas où l'objet Parametres n'aurait pas encore ces getters spécifiques
        try:
            self.coupon = parameters.get_coupon()
        except:
            self.coupon = getattr(parameters, 'coupon', 0.0)

        try:
            self.nominal = parameters.get_nominal()
        except:
            self.nominal = getattr(parameters, 'nominal', 100.0)
            
        self.maturity = parameters.get_maturity()
        self.rate = parameters.get_risk_free_rate()
        
        self.parameters = parameters

    def pricing_bond(self):
        c = self.coupon
        r = self.rate
        T = self.maturity
        N = self.nominal

        # Pricing flux actualisés
        # Coupons annuels, datés à rebours depuis la maturité : T, T-1, T-2, ... (> 0)
        # - T entier (ex: 5)   -> coupons en 1, 2, 3, 4, 5
        # - T non entier (5.5) -> coupons en 0.5, 1.5, ..., 5.5
        # Le dernier coupon est versé avec le nominal en T : il ne doit être compté qu'une fois.
        price = 0
        t = T - 1
        while t > 1e-9:
            price += c / (1+r)**t
            t -= 1

        # Flux final (Nominal + dernier coupon) actualisé à la date exacte T
        # Si Zéro-coupon (c=0), cela revient à N / (1+r)^T
        price += (c + N) / (1+r)**T
        return price

    # --- AJOUTS OBLIGATOIRES POUR STRUCTURED PRODUCT ---

    def payoff(self, spot_at_maturity: float) -> float:
        """
        Payoff à maturité.
        Un Bond paie son Nominal fixe, quel que soit le prix de l'action.
        """
        return self.nominal

    def delta(self) -> float:
        """Pas de sensibilité au prix de l'action."""
        return 0.0

    def vega(self) -> float:
        """Pas de sensibilité à la volatilité."""
        return 0.0