from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
import pandas as pd


# ----------------------------
# Optional: tiny helper struct
# ----------------------------
@dataclass
class StrategyParams:
    name: str
    legs: List[Dict[str, Any]]  # each has: instrument, quantity, model, label


class StructuredProduct:
    """
    Generic structured product: weighted sum of legs
    P_struct = sum_i quantity_i * Price_i

    Compatible with the existing code-1 architecture:
    - each leg holds: instrument, quantity, model, label
    - model may have price_option(inst), delta(inst), vega(inst)
    - instrument may have pricing_bond(), pricing_future(), payoff(spot)
    """

    def __init__(self, name: str = "Produit Structuré"):
        self.name = name
        self.legs: List[Dict[str, Any]] = []

    # ----------------------------
    # Core generic API (unchanged)
    # ----------------------------
    def add_leg(self, instrument, quantity: float, model=None, label: str = ""):
        self.legs.append({
            "instrument": instrument,
            "quantity": float(quantity),
            "model": model,
            "label": label
        })

    def price(self) -> float:
        total_price = 0.0

        for leg in self.legs:
            inst = leg["instrument"]
            qty = leg["quantity"]
            model = leg["model"]

            price_leg = 0.0

            # 1) Option-like: model.price_option(inst)
            if model is not None and hasattr(model, "price_option"):
                price_leg = model.price_option(inst)

            # 2) Bond-like: inst.pricing_bond()
            elif hasattr(inst, "pricing_bond"):
                price_leg = inst.pricing_bond()

            # 3) Future-like: inst.pricing_future()
            elif hasattr(inst, "pricing_future"):
                price_leg = inst.pricing_future()

            else:
                raise TypeError(
                    f"Cannot price leg '{leg.get('label','')}'. "
                    f"Provide a model with price_option(inst) or an instrument with pricing_bond()/pricing_future()."
                )

            total_price += price_leg * qty

        return float(total_price)

    def payoff(self, spot_at_maturity: float) -> float:
        total_payoff = 0.0

        for leg in self.legs:
            inst = leg["instrument"]
            qty = leg["quantity"]

            if hasattr(inst, "payoff"):
                total_payoff += inst.payoff(spot_at_maturity) * qty

            # Bond fallback: if you have a Bond.parameters.nominal
            elif hasattr(inst, "parameters") and hasattr(inst.parameters, "nominal"):
                total_payoff += float(inst.parameters.nominal) * qty

            else:
                raise TypeError(
                    f"Cannot compute payoff for leg '{leg.get('label','')}'. "
                    f"Instrument must implement payoff(spot) (or be a Bond with parameters.nominal)."
                )

        return float(total_payoff)

    def compute_greeks(self, spot_bump: float = 0.01, vol_bump: float = 0.01) -> Dict[str, float]:
        # NOTE: uses existing delta/vega methods if present (like code 1)
        total_delta = 0.0
        total_vega = 0.0

        for leg in self.legs:
            inst = leg["instrument"]
            qty = leg["quantity"]
            model = leg["model"]

            if model is not None and hasattr(model, "delta"):
                total_delta += model.delta(inst) * qty
            elif hasattr(inst, "delta"):
                total_delta += inst.delta() * qty

            if model is not None and hasattr(model, "vega"):
                total_vega += model.vega(inst) * qty
            elif hasattr(inst, "vega"):
                total_vega += inst.vega() * qty

        return {"Delta": float(total_delta), "Vega": float(total_vega)}

    def generate_payoff_profile(self, spot_range: np.ndarray) -> pd.DataFrame:
        payoffs = [self.payoff(float(s)) for s in spot_range]
        return pd.DataFrame({"Spot": spot_range, "Payoff Produit": payoffs})

    # ----------------------------------------
    # NEW: features from code 2 (safe additions)
    # ----------------------------------------

    @staticmethod
    def _validate_strikes(
        strategy: str,
        S0: Optional[float],
        strikes: Dict[str, float],
        strict: bool = True
    ) -> None:
        """
        Replicates code-2 style constraints, but optionally.
        If S0 is None or strict=False, does nothing.
        """
        if S0 is None or not strict:
            return

        s = float(S0)

        if strategy == "Bull call spread":
            if not (strikes["K1"] < s):
                raise ValueError(f"Bull call spread requires K1 < S0 (K1={strikes['K1']}, S0={s}).")
            if not (strikes["K2"] > s):
                raise ValueError(f"Bull call spread requires K2 > S0 (K2={strikes['K2']}, S0={s}).")
            if not (strikes["K2"] > strikes["K1"]):
                raise ValueError(f"Bull call spread requires K2 > K1 (K1={strikes['K1']}, K2={strikes['K2']}).")

        elif strategy == "Bear put spread":
            if not (strikes["K1"] > s):
                raise ValueError(f"Bear put spread requires K1 > S0 (K1={strikes['K1']}, S0={s}).")
            if not (strikes["K2"] < s):
                raise ValueError(f"Bear put spread requires K2 < S0 (K2={strikes['K2']}, S0={s}).")
            if not (strikes["K1"] > strikes["K2"]):
                raise ValueError(f"Bear put spread requires K1 > K2 (K1={strikes['K1']}, K2={strikes['K2']}).")

        elif strategy == "Long straddle":
            # no constraint except strike is positive
            if strikes["K"] <= 0:
                raise ValueError("Long straddle requires K > 0.")

        elif strategy == "Butterfly":
            K1, K2, K3 = strikes["K1"], strikes["K2"], strikes["K3"]
            if not (K1 < K2 < K3):
                raise ValueError(f"Butterfly requires K1 < K2 < K3 (got {K1}, {K2}, {K3}).")

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @classmethod
    def from_strategy(
        cls,
        strategy: str,
        *,
        # instruments factory: you provide how to build an instrument given (type_option, K)
        make_option_instrument: Callable[[str, float], Any],
        # model factory: you provide how to build a pricing model for this instrument
        make_option_model: Callable[[Any], Any],
        # strikes
        K1: Optional[float] = None,
        K2: Optional[float] = None,
        K3: Optional[float] = None,
        K: Optional[float] = None,
        # spot for validations
        S0: Optional[float] = None,
        validate_vs_spot: bool = True,
        name: Optional[str] = None,
    ) -> "StructuredProduct":
        """
        Adds code-2 "strategy preset" feature while keeping code-1 generic legs.

        You *must* pass two callables compatible with your existing structure:
        - make_option_instrument(type_option, K) -> option object that has payoff(spot)
        - make_option_model(option_obj) -> model object with price_option(option_obj) (and maybe greeks)

        This avoids hardcoding Pricing_option_BS vs BlackScholes etc.
        """
        strategy = strategy.strip()

        if name is None:
            name = f"Produit Structuré - {strategy}"

        sp = cls(name=name)

        if strategy == "Bull call spread":
            if K1 is None or K2 is None:
                raise ValueError("Bull call spread requires K1 and K2.")
            strikes = {"K1": float(K1), "K2": float(K2)}
            cls._validate_strikes(strategy, S0, strikes, strict=validate_vs_spot)

            # Standard: long call K1, short call K2
            opt_long = make_option_instrument("call", strikes["K1"])
            opt_short = make_option_instrument("call", strikes["K2"])
            sp.add_leg(opt_long, +1.0, make_option_model(opt_long), label=f"Long Call K={strikes['K1']}")
            sp.add_leg(opt_short, -1.0, make_option_model(opt_short), label=f"Short Call K={strikes['K2']}")

        elif strategy == "Bear put spread":
            if K1 is None or K2 is None:
                raise ValueError("Bear put spread requires K1 and K2.")
            strikes = {"K1": float(K1), "K2": float(K2)}
            cls._validate_strikes(strategy, S0, strikes, strict=validate_vs_spot)

            # Standard: long put K1 (higher strike), short put K2 (lower strike)
            opt_long = make_option_instrument("put", strikes["K1"])
            opt_short = make_option_instrument("put", strikes["K2"])
            sp.add_leg(opt_long, +1.0, make_option_model(opt_long), label=f"Long Put K={strikes['K1']}")
            sp.add_leg(opt_short, -1.0, make_option_model(opt_short), label=f"Short Put K={strikes['K2']}")

        elif strategy == "Long straddle":
            if K is None:
                raise ValueError("Long straddle requires K.")
            strikes = {"K": float(K)}
            cls._validate_strikes(strategy, S0, strikes, strict=validate_vs_spot)

            opt_call = make_option_instrument("call", strikes["K"])
            opt_put = make_option_instrument("put", strikes["K"])
            sp.add_leg(opt_call, +1.0, make_option_model(opt_call), label=f"Long Call K={strikes['K']}")
            sp.add_leg(opt_put, +1.0, make_option_model(opt_put), label=f"Long Put K={strikes['K']}")

        elif strategy == "Butterfly":
            if K1 is None or K2 is None or K3 is None:
                raise ValueError("Butterfly requires K1, K2, K3.")
            strikes = {"K1": float(K1), "K2": float(K2), "K3": float(K3)}
            cls._validate_strikes(strategy, S0, strikes, strict=validate_vs_spot)

            # Standard call butterfly: +1 C(K1) -2 C(K2) +1 C(K3)
            opt_k1 = make_option_instrument("call", strikes["K1"])
            opt_k2a = make_option_instrument("call", strikes["K2"])
            opt_k2b = make_option_instrument("call", strikes["K2"])
            opt_k3 = make_option_instrument("call", strikes["K3"])

            sp.add_leg(opt_k1, +1.0, make_option_model(opt_k1), label=f"Long Call K={strikes['K1']}")
            sp.add_leg(opt_k2a, -1.0, make_option_model(opt_k2a), label=f"Short Call K={strikes['K2']} (1)")
            sp.add_leg(opt_k2b, -1.0, make_option_model(opt_k2b), label=f"Short Call K={strikes['K2']} (2)")
            sp.add_leg(opt_k3, +1.0, make_option_model(opt_k3), label=f"Long Call K={strikes['K3']}")

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return sp

    def describe_legs(self) -> pd.DataFrame:
        """
        Convenience: list legs like an app would display.
        """
        rows = []
        for i, leg in enumerate(self.legs, start=1):
            inst = leg["instrument"]
            model = leg["model"]
            rows.append({
                "leg": i,
                "label": leg.get("label", ""),
                "quantity": leg["quantity"],
                "instrument_type": type(inst).__name__,
                "model_type": type(model).__name__ if model is not None else None,
            })
        return pd.DataFrame(rows)

    def price_client_view(self) -> float:
        """
        Code-2 idea: interpret sign as 'client cashflow at inception'.
        With our convention:
        - long legs (+qty) cost money => positive price() means client pays
        - short legs (-qty) receives money => negative contribution

        So we simply return self.price() and you can interpret it in your UI.
        If you prefer the opposite sign in your app, return -self.price().
        """
        return self.price()


# ----------------------------
# Example wiring (YOU adapt)
# ----------------------------
"""
You must plug this into your own classes.

Example idea (pseudo):

from modele.option_vanille import OptionVanille
from modele.black_scholes_model import BlackScholesModel

def make_inst(type_opt, K):
    return OptionVanille(K=K, type_option=type_opt, ...)   # must have payoff(spot)

def make_model(inst):
    return BlackScholesModel(S0=..., r=..., T=..., sigma=...)  # must have price_option(inst)

sp = StructuredProduct.from_strategy(
    "Bull call spread",
    make_option_instrument=make_inst,
    make_option_model=make_model,
    K1=180, K2=220,
    S0=200,
)

print(sp.describe_legs())
print("Price:", sp.price_client_view())
df_payoff = sp.generate_payoff_profile(np.linspace(0.5*S0, 1.5*S0, 200))
"""
