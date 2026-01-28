from src.utils.parametres import Parametres
import numpy as np

class BarrierOption:
    def __init__(self, parameters: Parametres, spot: float, option_type: str = "call", 
                 barrier: float = 80.0, barrier_type: str = "down-out"):
        """
        Args:
            barrier (float): Niveau de la barrière.
            barrier_type (str): 'down-out', 'down-in', 'up-out', 'up-in'.
        """
        self.spot = spot
        self.option_type = option_type.lower()
        self.parameters = parameters
        self.barrier = barrier
        self.barrier_type = barrier_type.lower()