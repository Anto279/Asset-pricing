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

        # Pricing flux actualisés (Ta logique)
        # Note : Si T est float (ex: 0.5 an), la boucle range(1, int(0.5)) ne s'exécute pas.
        # C'est correct pour un Zéro-Coupon (pas de flux intermédiaires).
        price = 0
        for t in range(1, int(T) + 1): # +1 pour inclure l'année T si T est entier
             price += c / (1+r)**t
             
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