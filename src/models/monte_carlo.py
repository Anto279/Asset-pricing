import numpy as np
from src.instruments.option_vanille import OptionVanille
from src.utils.parametres import Parametres


class MonteCarloModel:
    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = int(num_simulations)

    # --- small robustness: percent -> decimal if needed ---
    @staticmethod
    def _to_decimal(x: float) -> float:
        x = float(x)
        return x / 100.0 if abs(x) > 1.0 else x

    @staticmethod
    def _get_q(parameters) -> float:
        """
        Try common names for dividend yield. Defaults to 0.0 if not found.
        Supports your Parametres: dividend + get_dividend()
        """
        for name in ("get_dividend_yield", "get_q", "get_dividend", "dividend", "dividend_yield", "q"):
            if hasattr(parameters, name):
                val = getattr(parameters, name)
                out = val() if callable(val) else val
                return 0.0 if out is None else float(out)
        return 0.0

    def price_option(self, option: OptionVanille, seed: int | None = None) -> float:
        rng = np.random.default_rng(seed)

        opt_type = str(option.option).lower()
        S0 = float(option.spot)

        r = self._to_decimal(option.parameters.get_risk_free_rate())
        T = float(option.parameters.get_maturity())
        K = float(option.parameters.get_strike())
        sigma = self._to_decimal(option.parameters.get_volatility())
        q = self._to_decimal(self._get_q(option.parameters))

        if T <= 0:
            if opt_type == "call":
                return max(S0 - K, 0.0)
            if opt_type == "put":
                return max(K - S0, 0.0)
            return 0.0

        Z = rng.standard_normal(self.num_simulations)
        ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

        if opt_type == "call":
            payoffs = np.maximum(ST - K, 0.0)
        elif opt_type == "put":
            payoffs = np.maximum(K - ST, 0.0)
        else:
            return 0.0

        return float(np.exp(-r * T) * np.mean(payoffs))

    def Greeks_diff_MC(self, option: OptionVanille, h: float | None = None):
        if h is None:
            h = float(option.spot) * 0.01

        fixed_seed = 42
        price_central = self.price_option(option, seed=fixed_seed)

        option_up = OptionVanille(option.parameters, spot=option.spot + h, option=option.option)
        price_up = self.price_option(option_up, seed=fixed_seed)

        option_down = OptionVanille(option.parameters, spot=option.spot - h, option=option.option)
        price_down = self.price_option(option_down, seed=fixed_seed)

        delta = (price_up - price_down) / (2 * h)
        gamma = (price_up - 2 * price_central + price_down) / (h ** 2)

        # Vega bump (0.001 = 10 bps vol)
        h_vol = 0.001
        q0 = self._get_q(option.parameters)

        new_params = Parametres(
            maturity=option.parameters.get_maturity(),
            risk_free_rate=option.parameters.get_risk_free_rate(),
            strike=option.parameters.get_strike(),
            volatility=option.parameters.get_volatility() + h_vol,
            dividend=q0,  # ✅ preserve dividend yield
        )

        option_vega_up = OptionVanille(new_params, spot=option.spot, option=option.option)
        price_vega_up = self.price_option(option_vega_up, seed=fixed_seed)

        vega = (price_vega_up - price_central) / h_vol
        return float(delta), float(gamma), float(vega)

    def simulate_price_paths(
        self,
        S0: float,
        r: float,
        sigma: float,
        T: float,
        steps: int,
        q: float = 0.0,
        n_paths: int | None = None,
        seed: int | None = None,
    ):
        """
        Simulate GBM paths for visualization (not pricing).
        n_paths defaults to min(self.num_simulations, 5000) to avoid huge memory usage.
        """
        steps = int(steps)
        if steps <= 0:
            return np.array([[float(S0)]], dtype=float)

        n = int(n_paths) if n_paths is not None else int(min(self.num_simulations, 5000))

        r = self._to_decimal(r)
        sigma = self._to_decimal(sigma)
        q = self._to_decimal(q)

        rng = np.random.default_rng(seed)

        dt = float(T) / steps
        paths = np.zeros((n, steps + 1), dtype=float)
        paths[:, 0] = float(S0)

        drift = (r - q - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)

        for t in range(1, steps + 1):
            Z = rng.standard_normal(n)
            paths[:, t] = paths[:, t - 1] * np.exp(drift + vol * Z)

        return paths
