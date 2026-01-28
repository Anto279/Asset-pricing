import numpy as np

class Parametres:
    """
    Classe contenant les paramètres financiers d'un produit.
    """
    def __init__(
        self,
        coupon: float = None,
        maturity: float = None,
        nominal: float = None,
        risk_free_rate: float = None,
        frequency: int = None,
        dividend: float = None,     # q
        strike: float = None,
        volatility: float = None
    ):
        self.coupon = coupon
        self.maturity = maturity
        self.nominal = nominal
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        self.dividend = dividend
        self.strike = strike
        self.volatility = volatility

    def get_coupon(self) -> float:
        return self.coupon

    def get_maturity(self) -> float:
        return self.maturity

    def get_nominal(self) -> float:
        return self.nominal

    def get_risk_free_rate(self) -> float:
        return self.risk_free_rate

    def get_frequency(self) -> int:
        return self.frequency

    # Make this safe (never None)
    def get_dividend(self) -> float:
        return 0.0 if self.dividend is None else float(self.dividend)

    def get_dividend_yield(self) -> float:
        return self.get_dividend()

    def get_q(self) -> float:
        return self.get_dividend()

    def get_strike(self) -> float:
        return self.strike

    def get_volatility(self) -> float:
        return self.volatility

    def get_actualisation(self, r, T) -> float:
        return np.exp(-r * T)
