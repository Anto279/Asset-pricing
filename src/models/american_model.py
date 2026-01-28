import numpy as np


class AmericanBinomialTreeModel:
    """
    Cox-Ross-Rubinstein (CRR) binomial tree.
    Supports European (american=False) and American (american=True) via early exercise.
    Handles continuous dividend yield q if provided by the option/parameters.
    """

    def __init__(self, steps: int = 500):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.steps = int(steps)

    # --------- robust getters (works with your OptionVanille + Parametres) ---------
    @staticmethod
    def _get_from(obj, names, default=None):
        for name in names:
            if hasattr(obj, name):
                val = getattr(obj, name)
                # support getters like get_strike()
                if callable(val) and name.startswith("get_"):
                    return val()
                return val
        return default

    def _extract_inputs(self, option):
        # spot
        S0 = self._get_from(option, ["spot", "S0", "S"], None)
        if S0 is None:
            raise AttributeError("Cannot find spot on option (expected option.spot).")
        S0 = float(S0)

        # parameters container
        params = self._get_from(option, ["parameters", "params"], None)

        # strike, maturity, r, sigma, q
        K = self._get_from(params, ["strike", "K", "get_strike"], None)
        T = self._get_from(params, ["maturity", "T", "get_maturity"], None)
        r = self._get_from(params, ["risk_free_rate", "r", "get_risk_free_rate"], None)
        sigma = self._get_from(params, ["volatility", "sigma", "get_volatility"], None)
        q = self._get_from(params, ["dividend", "dividend_yield", "q", "get_dividend"], 0.0)

        if K is None or T is None or r is None or sigma is None:
            raise AttributeError("Missing one of strike/maturity/risk_free_rate/volatility in option.parameters.")

        # option type
        opt_type = self._get_from(option, ["option", "option_type", "type_option"], None)
        if opt_type is None:
            raise AttributeError("Cannot find option type (expected option.option == 'call'/'put').")
        opt_type = str(opt_type).lower()

        if opt_type not in ("call", "put"):
            raise ValueError(f"Unknown option type: {opt_type}")

        return float(S0), float(K), float(T), float(r), float(sigma), float(q), opt_type

    # --------- payoff helpers ---------
    @staticmethod
    def _intrinsic(opt_type: str, S: np.ndarray, K: float) -> np.ndarray:
        if opt_type == "call":
            return np.maximum(S - K, 0.0)
        return np.maximum(K - S, 0.0)

    # --------- pricing ---------
    def price_option(self, option, american: bool = True) -> float:
        S0, K, T, r, sigma, q, opt_type = self._extract_inputs(option)

        N = self.steps
        dt = T / N
        if dt <= 0:
            # maturity ~ 0 : immediate exercise
            return float(self._intrinsic(opt_type, np.array([S0]), K)[0])

        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u

        disc = np.exp(-r * dt)
        a = np.exp((r - q) * dt)
        p = (a - d) / (u - d)

        # numeric sanity
        if not (0.0 <= p <= 1.0):
            # If extreme params cause p outside [0,1], tree becomes unstable.
            # We clamp softly rather than crash (better UX in Streamlit).
            p = min(max(p, 0.0), 1.0)

        # terminal underlying prices
        j = np.arange(N + 1)
        S_T = S0 * (u ** j) * (d ** (N - j))

        # terminal option values
        V = self._intrinsic(opt_type, S_T, K)

        # backward induction
        for i in range(N - 1, -1, -1):
            # continuation values
            V = disc * (p * V[1:] + (1.0 - p) * V[:-1])

            if american:
                # compute intrinsic at time i for nodes j=0..i
                j = np.arange(i + 1)
                S_i = S0 * (u ** j) * (d ** (i - j))
                exercise = self._intrinsic(opt_type, S_i, K)
                V = np.maximum(V, exercise)

        return float(V[0])


class LongstaffSchwartzMonteCarloModel:
    """
    American option pricing via Longstaff-Schwartz (Least-Squares Monte Carlo).
    Works well for American puts; for calls without dividends, early exercise is typically not optimal.
    """

    def __init__(self, n_paths: int = 50000, steps: int = 100, seed: int = 123, degree: int = 2):
        if n_paths < 1000:
            raise ValueError("n_paths too small (use >= 1000).")
        if steps < 2:
            raise ValueError("steps must be >= 2 for LSM.")
        if degree < 1:
            raise ValueError("degree must be >= 1.")
        self.n_paths = int(n_paths)
        self.steps = int(steps)
        self.seed = int(seed)
        self.degree = int(degree)

    # --------- robust getters ---------
    @staticmethod
    def _get_from(obj, names, default=None):
        for name in names:
            if hasattr(obj, name):
                val = getattr(obj, name)
                if callable(val) and name.startswith("get_"):
                    return val()
                return val
        return default

    def _extract_inputs(self, option):
        S0 = self._get_from(option, ["spot", "S0", "S"], None)
        if S0 is None:
            raise AttributeError("Cannot find spot on option (expected option.spot).")
        S0 = float(S0)

        params = self._get_from(option, ["parameters", "params"], None)

        K = self._get_from(params, ["strike", "K", "get_strike"], None)
        T = self._get_from(params, ["maturity", "T", "get_maturity"], None)
        r = self._get_from(params, ["risk_free_rate", "r", "get_risk_free_rate"], None)
        sigma = self._get_from(params, ["volatility", "sigma", "get_volatility"], None)
        q = self._get_from(params, ["dividend", "dividend_yield", "q", "get_dividend"], 0.0)

        if K is None or T is None or r is None or sigma is None:
            raise AttributeError("Missing one of strike/maturity/risk_free_rate/volatility in option.parameters.")

        opt_type = self._get_from(option, ["option", "option_type", "type_option"], None)
        if opt_type is None:
            raise AttributeError("Cannot find option type (expected option.option == 'call'/'put').")
        opt_type = str(opt_type).lower()
        if opt_type not in ("call", "put"):
            raise ValueError(f"Unknown option type: {opt_type}")

        return float(S0), float(K), float(T), float(r), float(sigma), float(q), opt_type

    @staticmethod
    def _intrinsic(opt_type: str, S: np.ndarray, K: float) -> np.ndarray:
        if opt_type == "call":
            return np.maximum(S - K, 0.0)
        return np.maximum(K - S, 0.0)

    def _simulate_gbm_paths(self, S0: float, T: float, r: float, q: float, sigma: float) -> np.ndarray:
        np.random.seed(self.seed)
        dt = T / self.steps
        drift = (r - q - 0.5 * sigma * sigma) * dt
        vol = sigma * np.sqrt(dt)

        Z = np.random.standard_normal((self.n_paths, self.steps))
        log_increments = drift + vol * Z

        S = np.empty((self.n_paths, self.steps + 1), dtype=float)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(np.cumsum(log_increments, axis=1))
        return S

    def _basis(self, x: np.ndarray) -> np.ndarray:
        # polynomial basis: [1, x, x^2, ..., x^degree]
        cols = [np.ones_like(x)]
        for k in range(1, self.degree + 1):
            cols.append(x ** k)
        return np.column_stack(cols)

    def price_option(self, option) -> float:
        S0, K, T, r, sigma, q, opt_type = self._extract_inputs(option)
        if T <= 0:
            return float(self._intrinsic(opt_type, np.array([S0]), K)[0])

        S = self._simulate_gbm_paths(S0, T, r, q, sigma)
        dt = T / self.steps
        disc = np.exp(-r * dt)

        # cashflow at maturity
        cf = self._intrinsic(opt_type, S[:, -1], K)

        # backward induction in cashflows (discounting step by step)
        for t in range(self.steps - 1, 0, -1):
            cf *= disc  # discount from t+1 to t

            St = S[:, t]
            intrinsic = self._intrinsic(opt_type, St, K)

            itm = intrinsic > 0.0
            if not np.any(itm):
                continue

            X = self._basis(St[itm])
            y = cf[itm]

            # least squares regression: continuation value
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            continuation = X @ beta

            exercise_now = intrinsic[itm] > continuation
            # replace cf with intrinsic where exercise is optimal
            y[exercise_now] = intrinsic[itm][exercise_now]
            cf[itm] = y

        # discount from t=1 to t=0
        price = float(np.mean(cf) * disc)
        return price
