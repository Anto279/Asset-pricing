from src.utils.parametres import Parametres
import pandas as pd

class OptionVanille:
    def __init__(self, parameters : Parametres , spot : float, option : str = "call"):
        self.spot = spot
        self.option = option
        self.parameters = parameters

    def payoff(self, spot_at_maturity : float):
        K = self.parameters.get_strike()
        if self.option == "call":
            return max(spot_at_maturity - K, 0)
        elif self.option == "put":
            return max(K - spot_at_maturity, 0)
        
    def payoff_graph(self):
        K = self.parameters.get_strike()
        spots = [i for i in range(0, int(K*2)+1)]
        payoffs = []
        for S in spots:
            payoffs.append(self.payoff(S))
        data_payoff = {"spots": spots, "payoffs": payoffs}
        data_payoff = pd.DataFrame(data_payoff)
        return data_payoff
    

    