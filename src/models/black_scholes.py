import numpy as np
from scipy.stats import norm
from src.instruments.option_vanille import OptionVanille

class BlackScholesModel:
    """
    Black-Scholes with continuous dividend yield q (a.k.a. cost of carry b = r - q).
    - Supports scalar or numpy-array spot (so your compute_greeks_range works).
    - Adds a small "percent -> decimal" guardrail (heuristic).
    """

    def __init__(self):
        pass

    # ---------- helpers ----------
    @staticmethod
    def _to_decimal(x: float) -> float:
        """
        Heuristic: if x looks like a percent (e.g. 3 or 20), convert to decimal (0.03 / 0.20).
        If you already pass decimals everywhere, this won't change typical values (<1).
        """
        x = float(x)
        return x / 100.0 if abs(x) > 1.0 else x

    @staticmethod
    def _get_q(params) -> float:
        # Try common names/getters for dividend yield
        for name in ("get_dividend_yield", "get_dividend", "get_q"):
            if hasattr(params, name) and callable(getattr(params, name)):
                return float(getattr(params, name)())
        for attr in ("dividend_yield", "dividend", "q"):
            if hasattr(params, attr):
                return float(getattr(params, attr))
        return 0.0

    @staticmethod
    def _d1_d2(S, K, T, r, q, sigma):
        # S may be scalar or ndarray
        S = np.asarray(S, dtype=float)
        if T <= 0 or sigma <= 0:
            # Not meaningful to compute d1/d2; caller should handle separately
            return np.full_like(S, np.nan), np.full_like(S, np.nan)

        vol_sqrtT = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrtT
        d2 = d1 - vol_sqrtT
        return d1, d2

    # ---------- pricing ----------
    def price_option(self, option: OptionVanille):
        type_option = str(option.option).lower()
        S = option.spot
        params = option.parameters

        r = self._to_decimal(params.get_risk_free_rate())
        T = float(params.get_maturity())
        K = float(params.get_strike())
        sigma = self._to_decimal(params.get_volatility())
        q = self._to_decimal(self._get_q(params))

        # Handle expiry / degenerate vol
        if T <= 0:
            if type_option == "call":
                return float(np.maximum(np.asarray(S, dtype=float) - K, 0.0))
            elif type_option == "put":
                return float(np.maximum(K - np.asarray(S, dtype=float), 0.0))
            raise ValueError("option.option must be 'call' or 'put'.")

        if sigma <= 0:
            # Intrinsic of forward-discounted payoff bound (simple fallback)
            # (If you want a more rigorous sigma->0 limit, we can do it too.)
            if type_option == "call":
                return float(np.maximum(np.asarray(S, dtype=float) * np.exp(-q*T) - K*np.exp(-r*T), 0.0))
            elif type_option == "put":
                return float(np.maximum(K*np.exp(-r*T) - np.asarray(S, dtype=float) * np.exp(-q*T), 0.0))
            raise ValueError("option.option must be 'call' or 'put'.")

        d1, d2 = self._d1_d2(S, K, T, r, q, sigma)
        S = np.asarray(S, dtype=float)

        if type_option == "call":
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif type_option == "put":
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            raise ValueError("option.option must be 'call' or 'put'.")

        return float(price) if np.ndim(price) == 0 else price

    # ---------- greeks ----------
    def delta(self, option: OptionVanille):
        type_option = str(option.option).lower()
        S = option.spot
        params = option.parameters

        r = self._to_decimal(params.get_risk_free_rate())
        T = float(params.get_maturity())
        K = float(params.get_strike())
        sigma = self._to_decimal(params.get_volatility())
        q = self._to_decimal(self._get_q(params))

        if T <= 0 or sigma <= 0:
            # At expiry: delta is not well-defined at-the-money; use a simple convention
            S_arr = np.asarray(S, dtype=float)
            if type_option == "call":
                out = (S_arr > K).astype(float)
            else:
                out = -(S_arr < K).astype(float)
            return float(out) if np.ndim(out) == 0 else out

        d1, _ = self._d1_d2(S, K, T, r, q, sigma)
        disc_q = np.exp(-q * T)

        if type_option == "call":
            out = disc_q * norm.cdf(d1)
        elif type_option == "put":
            out = disc_q * (norm.cdf(d1) - 1.0)
        else:
            raise ValueError("option.option must be 'call' or 'put'.")

        return float(out) if np.ndim(out) == 0 else out

    def gamma(self, option: OptionVanille):
        S = np.asarray(option.spot, dtype=float)
        params = option.parameters

        r = self._to_decimal(params.get_risk_free_rate())
        T = float(params.get_maturity())
        K = float(params.get_strike())
        sigma = self._to_decimal(params.get_volatility())
        q = self._to_decimal(self._get_q(params))

        if T <= 0 or sigma <= 0:
            out = np.zeros_like(S, dtype=float)
            return float(out) if out.ndim == 0 else out

        d1, _ = self._d1_d2(S, K, T, r, q, sigma)
        out = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

        return float(out) if out.ndim == 0 else out

    def vega(self, option: OptionVanille):
        S = np.asarray(option.spot, dtype=float)
        params = option.parameters

        r = self._to_decimal(params.get_risk_free_rate())
        T = float(params.get_maturity())
        K = float(params.get_strike())
        sigma = self._to_decimal(params.get_volatility())
        q = self._to_decimal(self._get_q(params))

        if T <= 0 or sigma <= 0:
            out = np.zeros_like(S, dtype=float)
            return float(out) if out.ndim == 0 else out

        d1, _ = self._d1_d2(S, K, T, r, q, sigma)
        out = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

        return float(out) if out.ndim == 0 else out

    def rho(self, option: OptionVanille):
        type_option = str(option.option).lower()
        S = option.spot
        params = option.parameters

        r = self._to_decimal(params.get_risk_free_rate())
        T = float(params.get_maturity())
        K = float(params.get_strike())
        sigma = self._to_decimal(params.get_volatility())
        q = self._to_decimal(self._get_q(params))

        if T <= 0 or sigma <= 0:
            return 0.0

        _, d2 = self._d1_d2(S, K, T, r, q, sigma)
        if type_option == "call":
            out = K * T * np.exp(-r * T) * norm.cdf(d2)
        elif type_option == "put":
            out = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        else:
            raise ValueError("option.option must be 'call' or 'put'.")

        return float(out) if np.ndim(out) == 0 else out

    def theta(self, option: OptionVanille):
        """
        Returns theta per YEAR (same convention as many textbooks).
        If you want per day: theta / 365.
        """
        type_option = str(option.option).lower()
        S = np.asarray(option.spot, dtype=float)
        params = option.parameters

        r = self._to_decimal(params.get_risk_free_rate())
        T = float(params.get_maturity())
        K = float(params.get_strike())
        sigma = self._to_decimal(params.get_volatility())
        q = self._to_decimal(self._get_q(params))

        if T <= 0 or sigma <= 0:
            out = np.zeros_like(S, dtype=float)
            return float(out) if out.ndim == 0 else out

        d1, d2 = self._d1_d2(S, K, T, r, q, sigma)

        term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))

        if type_option == "call":
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            term3 = +q * S * np.exp(-q * T) * norm.cdf(d1)
            out = term1 + term2 + term3
        elif type_option == "put":
            term2 = +r * K * np.exp(-r * T) * norm.cdf(-d2)
            term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
            out = term1 + term2 + term3
        else:
            raise ValueError("option.option must be 'call' or 'put'.")

        return float(out) if out.ndim == 0 else out

    def compute_greeks_range(self, option: OptionVanille, range_pct=0.5, steps=100):
        K = option.parameters.get_strike()
        spot_values = np.linspace(K * (1 - range_pct), K * (1 + range_pct), steps)

        original_spot = option.spot
        option.spot = spot_values

        data = {
            "S": spot_values,
            "Delta": self.delta(option),
            "Gamma": self.gamma(option),
            "Vega": self.vega(option),
            "Theta": self.theta(option),
            "Rho": self.rho(option),
        }

        option.spot = original_spot
        return data
