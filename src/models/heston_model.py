# src/models/heston_model.py

from __future__ import annotations
import numpy as np
import pandas as pd


class HestonMonteCarloModel:
    """
    Heston stochastic volatility model (risk-neutral):
        dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW1_t
        dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2_t
        corr(dW1, dW2) = rho

    Discretization:
      - log-Euler for S (keeps S > 0)
      - Euler for v with "full truncation" to enforce v >= 0

    Pricing:
      European call/put via Monte Carlo.
    """

    def __init__(self, chunk_size: int = 20000):
        self.chunk_size = int(chunk_size)

    @staticmethod
    def _validate_inputs(
        S0: float, K: float, T: float, r: float, q: float,
        v0: float, kappa: float, theta: float, xi: float, rho: float,
        n_paths: int, steps: int, option_type: str
    ) -> None:
        if S0 <= 0:
            raise ValueError("S0 must be > 0.")
        if K <= 0:
            raise ValueError("K must be > 0.")
        if T <= 0:
            raise ValueError("T must be > 0.")
        if n_paths <= 0 or steps <= 0:
            raise ValueError("n_paths and steps must be > 0.")
        if v0 < 0 or theta < 0:
            raise ValueError("v0 and theta must be >= 0.")
        if kappa <= 0 or xi < 0:
            raise ValueError("kappa must be > 0 and xi must be >= 0.")
        if not (-0.999 <= rho <= 0.999):
            raise ValueError("rho must be in [-0.999, 0.999] to avoid numerical issues.")
        if option_type.lower() not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")

    def price_european_mc(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        q: float,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        option_type: str = "call",
        n_paths: int = 50000,
        steps: int = 252,
        seed: int | None = 123,
        antithetic: bool = True,
        full_truncation: bool = True,
    ) -> dict:
        """
        Returns a dict with:
          - price
          - std_error (Monte Carlo standard error)
          - ci_95 (tuple)
          - n_paths, steps
        """
        option_type = option_type.lower()
        self._validate_inputs(S0, K, T, r, q, v0, kappa, theta, xi, rho, n_paths, steps, option_type)

        dt = T / steps
        disc = float(np.exp(-r * T))

        rng = np.random.default_rng(seed)

        # accumulate payoffs without storing everything
        sum_payoff = 0.0
        sum_payoff_sq = 0.0
        done = 0

        # numeric safety
        rho = float(np.clip(rho, -0.999, 0.999))
        sqrt_1mr2 = float(np.sqrt(max(1.0 - rho * rho, 1e-12)))

        while done < n_paths:
            m = min(self.chunk_size, n_paths - done)

            # if antithetic, we generate half and mirror to reach m (best if m even)
            if antithetic:
                if m == 1:
                    m_eff = 1
                    use_anti = False
                else:
                    if m % 2 == 1:
                        m -= 1
                    m_eff = m // 2
                    use_anti = True
            else:
                m_eff = m
                use_anti = False

            if m_eff <= 0:
                # fallback in edge cases
                m = min(self.chunk_size, n_paths - done)
                m_eff = m
                use_anti = False

            Z1 = rng.standard_normal((m_eff, steps))
            Z2 = rng.standard_normal((m_eff, steps))

            if use_anti:
                Z1 = np.vstack([Z1, -Z1])
                Z2 = np.vstack([Z2, -Z2])

            # states
            logS = np.full(Z1.shape[0], np.log(S0), dtype=float)
            v = np.full(Z1.shape[0], v0, dtype=float)

            for t in range(steps):
                v_pos = np.maximum(v, 0.0)

                z1 = Z1[:, t]
                z2 = Z2[:, t]

                # correlated Brownian increments (scaled later with sqrt(dt))
                w1 = z1
                w2 = rho * z1 + sqrt_1mr2 * z2

                # log-Euler for S
                logS += (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * w1

                # Euler for v (full truncation drift/vol on v_pos)
                v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * w2

                if full_truncation:
                    v = np.maximum(v, 0.0)

            ST = np.exp(logS)

            if option_type == "call":
                payoff = np.maximum(ST - K, 0.0)
            else:
                payoff = np.maximum(K - ST, 0.0)

            sum_payoff += float(payoff.sum())
            sum_payoff_sq += float((payoff * payoff).sum())
            done += int(Z1.shape[0])

        mean_payoff = sum_payoff / done
        mean_payoff_sq = sum_payoff_sq / done
        var_payoff = max(mean_payoff_sq - mean_payoff * mean_payoff, 0.0)

        price = disc * mean_payoff
        std_error = disc * np.sqrt(var_payoff / done)
        ci_95 = (price - 1.96 * std_error, price + 1.96 * std_error)

        return {
            "price": float(price),
            "std_error": float(std_error),
            "ci_95": (float(ci_95[0]), float(ci_95[1])),
            "n_paths": int(done),
            "steps": int(steps),
        }

    def simulate_paths_for_visualization(
        self,
        S0: float,
        T: float,
        r: float,
        q: float,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        n_paths_visu: int = 30,
        steps: int = 252,
        seed: int | None = 123,
        full_truncation: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns:
          - df_S: index=time, columns=paths (S paths)
          - df_v: index=time, columns=paths (variance paths)
        """
        self._validate_inputs(S0, 1.0, T, r, q, v0, kappa, theta, xi, rho, n_paths_visu, steps, "call")

        dt = T / steps
        times = np.linspace(0.0, T, steps + 1)

        rng = np.random.default_rng(seed)
        rho = float(np.clip(rho, -0.999, 0.999))
        sqrt_1mr2 = float(np.sqrt(max(1.0 - rho * rho, 1e-12)))

        Z1 = rng.standard_normal((n_paths_visu, steps))
        Z2 = rng.standard_normal((n_paths_visu, steps))

        logS = np.full((n_paths_visu,), np.log(S0), dtype=float)
        v = np.full((n_paths_visu,), v0, dtype=float)

        S_paths = np.empty((steps + 1, n_paths_visu), dtype=float)
        v_paths = np.empty((steps + 1, n_paths_visu), dtype=float)
        S_paths[0, :] = S0
        v_paths[0, :] = v0

        for t in range(steps):
            v_pos = np.maximum(v, 0.0)

            w1 = Z1[:, t]
            w2 = rho * Z1[:, t] + sqrt_1mr2 * Z2[:, t]

            logS += (r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * w1
            v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * w2

            if full_truncation:
                v = np.maximum(v, 0.0)

            S_paths[t + 1, :] = np.exp(logS)
            v_paths[t + 1, :] = v

        df_S = pd.DataFrame(S_paths, index=times, columns=[f"Path_{i+1}" for i in range(n_paths_visu)])
        df_v = pd.DataFrame(v_paths, index=times, columns=[f"Path_{i+1}" for i in range(n_paths_visu)])
        return df_S, df_v
