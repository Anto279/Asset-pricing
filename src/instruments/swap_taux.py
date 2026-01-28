from src.utils.parametres import Parametres
import numpy as np
import pandas as pd

class Swap_taux:
    def __init__(self, parameters: Parametres):
        
        self.frequency = parameters.get_frequency()
        self.rate = parameters.get_risk_free_rate()
        self.maturity = parameters.get_maturity()
        self.nominal = parameters.get_nominal()
        
        self.parameters = parameters

    def taux_swap(self, list_risk_free_rates : list = []) -> float:
        T = self.maturity
        f = self.frequency
        r = self.rate
        delta = 1/f
        number_period = int(np.floor(T * f))

        A = 1 - self.parameters.get_actualisation(r, T=T)
        B = 0

        # Si le swap a moins d'une période complète, S0 est indéfini ou nécessite un traitement spécial (non géré ici)
        if number_period == 0:
            return 0.0

        if list_risk_free_rates == []:
            for i in range(1, number_period + 1):
                B += delta * self.parameters.get_actualisation(r, T=delta*i)
        elif list_risk_free_rates != [] :
            if len(list_risk_free_rates) != number_period:
                raise ValueError("La longueur de la liste des taux sans risque ne correspond pas au nombre de périodes.")
            for i in range(number_period):
                r_i = list_risk_free_rates[i]
                B += delta * self.parameters.get_actualisation(r_i, T=delta*(i+1))

        if B == 0:
            raise ValueError("Division par zéro dans le calcul du prix du swap.")
        
        S0 = A / B
        return S0
    
    def valuation_swap_fixed(self, list_risk_free_rates : list = [], fixed_rate=None):

        f = self.frequency
        T = self.maturity
        r = self.rate
        N = self.nominal
        D = 0
        delta = 1/f
        number_period = int(np.floor(T * f))

        if fixed_rate is None:
            S0 = self.taux_swap(list_risk_free_rates=list_risk_free_rates)
        else:
            S0 = fixed_rate

        # Si le swap a moins d'une période complète, S0 est indéfini ou nécessite un traitement spécial (non géré ici)
        if number_period == 0:
            return 0.0

        if list_risk_free_rates == []:
            for i in range(1, number_period + 1):
                D += delta * self.parameters.get_actualisation(r, T=delta*i)
        elif list_risk_free_rates != [] :
            if len(list_risk_free_rates) != number_period:
                raise ValueError("La longueur de la liste des taux sans risque ne correspond pas au nombre de périodes.")
            for i in range(number_period):
                r_i = list_risk_free_rates[i]
                D += delta * self.parameters.get_actualisation(r_i, T=delta*(i+1))

        #On utilise ici la fomule du prix du swap pour le côté fixe
        # PV_fixed = N * S0 * somme(delta * B(0, Ti))

        valuation_fixed = N * S0 * D 
        return valuation_fixed
    
    def valuation_swap_floating(self, list_risk_free_rates : list = []):
        f = self.frequency
        T = self.maturity
        r = self.rate
        N = self.nominal
        valuation_floating = 0
        delta = 1/f
        number_period = int(np.floor(T * f))

        # Si le swap a moins d'une période complète, S0 est indéfini ou nécessite un traitement spécial (non géré ici)
        if number_period == 0:
            return 0.0

        valuation_floating = N * (1 - self.parameters.get_actualisation(r, T=delta*number_period))
    
        # On utilise ici la fomule du prix du swap pour le côté flottant
        # PV_float = N * (1 - B(0, Tn))
        return valuation_floating

    # On calcul maintenant la valeur du swap de taux, simplement en faisant la différence entre la valeur du côté flottant et la valeur du côté fixe
    def valuation_swap(self, list_risk_free_rates : list = [], fixed_rate=None):
        pv_fixed = self.valuation_swap_fixed(list_risk_free_rates, fixed_rate=fixed_rate)
        pv_floating = self.valuation_swap_floating(list_risk_free_rates)
        pv_swap = pv_floating - pv_fixed # Payer Swap (Receiver Swap serait l'inverse)
        return pv_swap
    

    # Implémentation des méthodes de calcul des sensibilités pour le swap de taux

    # équivalent Delta, DV01 : Sensibilité du prix du swap à une variation parallèle des taux d'intérêt
    # On utilisera la formule mathématique déterminer pour une variation d'un point de base : 
    # DV01 = -0.0001 * dV_swap/dr. Dérivée vectorielle en fonction des ri.
    # DV01 = -0.0001 * N * (S0 * somme((-ti) * delta * B(ri, 0, ti)) - tn * B(rn, 0, Tn))

    def dv01(self, list_risk_free_rates : list = [], fixed_rate=None):
        f = self.frequency
        T = self.maturity
        r = self.rate
        N = self.nominal
        delta = 1/f
        number_period = int(np.floor(T * f))

        if number_period == 0:
            return 0.0
        if fixed_rate is None:
            S0 = self.taux_swap(list_risk_free_rates=list_risk_free_rates)
        else:
            S0 = fixed_rate

        sum = 0

        if list_risk_free_rates == []:
            for i in range(1, number_period + 1):
                ti = delta * i
                B_ti = self.parameters.get_actualisation(r, T=ti)
                sum += (-ti) * delta * B_ti
            tn = delta * number_period
            B_Tn = self.parameters.get_actualisation(r, T=tn)
        else:
            if len(list_risk_free_rates) != number_period:
                raise ValueError("La longueur de la liste des taux sans risque ne correspond pas au nombre de périodes.")
            for i in range(number_period):
                ti = delta * (i + 1)
                B_ti = self.parameters.get_actualisation(list_risk_free_rates[i], T=ti)
                sum += (-ti) * delta * B_ti
            tn = delta * number_period
            B_Tn = self.parameters.get_actualisation(list_risk_free_rates[-1], T=tn)

        dv01 = 0.0001 * N * (tn * B_Tn - S0 * sum) # Payer Swap (Receiver Swap serait l'inverse)

        return dv01
    
    # Convexité du swap de taux
    # dérivée seconde du prix du swap total par rapport aux taux d'intérêt
    # Convexity = d²V_swap/dr²
    # Convexity = - N * (S0 * somme((ti²) * delta * B(ri, 0, ti)) - tn² * B(rn, 0, Tn))

    def convexity(self, list_risk_free_rates : list = [], fixed_rate=None):
        f = self.frequency
        T = self.maturity
        r = self.rate
        N = self.nominal
        delta = 1/f
        number_period = int(np.floor(T * f))

        if number_period == 0:
            return 0.0

        # Pour permettre le calcul de la convexité avec un taux fixe manuel (différente du taux de marché(au pair))

        if fixed_rate is None:
            S0 = self.taux_swap(list_risk_free_rates=list_risk_free_rates)
        else:
            S0 = fixed_rate

        sum = 0

        if list_risk_free_rates == []: # On suppose une courbe plate avec valeur manuelle du taux sans risque
            for i in range(1, number_period + 1):
                ti = delta * i
                B_ti = self.parameters.get_actualisation(r, T=ti)
                sum += (ti ** 2) * delta * B_ti
            tn = delta * number_period
            B_Tn = self.parameters.get_actualisation(r, T=tn)

        else:
            if len(list_risk_free_rates) != number_period:
                raise ValueError("La longueur de la liste des taux sans risque ne correspond pas au nombre de périodes.")
            for i in range(number_period):
                ti = delta * (i + 1)
                B_ti = self.parameters.get_actualisation(list_risk_free_rates[i], T=ti)
                sum += (ti ** 2) * delta * B_ti
            tn = delta * number_period
            B_Tn = self.parameters.get_actualisation(list_risk_free_rates[-1], T=tn)

        convexity = - N * (S0 * sum + (tn ** 2) * B_Tn)

        return convexity
    
    # Carry théorique du swap de taux
    # On dérive la valeur du swap par rapport au temps
    # Carry = dV_swap/dt
    # Carry = N * S0 * somme(ri * delta * B(0, ti)) + N * rn * B(0, Tn)
    # Ici, pas besoin de supposer une courbe plate, on utilise les taux fournis dans la liste

    def carry(self, list_risk_free_rates : list = [], fixed_rate=None):
        f = self.frequency
        T = self.maturity
        r = self.rate
        N = self.nominal
        delta = 1/f
        number_period = int(np.floor(T * f))

        if number_period == 0:
            return 0.0

        if fixed_rate is None:
            S0 = self.taux_swap(list_risk_free_rates=list_risk_free_rates)
        else:
            S0 = fixed_rate

        sum = 0

        if list_risk_free_rates == []:
            for i in range(1, number_period + 1):
                ti = delta * i
                r_i = r
                B_ti = self.parameters.get_actualisation(r_i, T=ti)
                sum += r_i * delta * B_ti
            tn = delta * number_period
            r_n = r
            B_Tn = self.parameters.get_actualisation(r_n, T=tn)
        else:
            if len(list_risk_free_rates) != number_period:
                raise ValueError("La longueur de la liste des taux sans risque ne correspond pas au nombre de périodes.")
            for i in range(number_period):
                ti = delta * (i + 1)
                r_i = list_risk_free_rates[i]
                B_ti = self.parameters.get_actualisation(r_i, T=ti)
                sum += r_i * delta * B_ti
            tn = delta * number_period
            r_n = list_risk_free_rates[-1]
            B_Tn = self.parameters.get_actualisation(r_n, T=tn)

        carry = - N * S0 * sum - N * r_n * B_Tn

        return carry
    
    # Affichage graphique des sensibilités du swap de taux
    # Evolution du DV01 et Convexité en fonction du taux r 
    # Evolution du Carry en fonction du temps à maturité T

    def sensitivities_range(self, list_risk_free_rates : list = [], fixed_rate=None, range_pct=0.5, steps=100):
        original_rate = self.rate
        data = {}

        if list_risk_free_rates == []:
            # Création de l'axe des X (les taux simulés)
            rate_values = np.linspace(max(original_rate * (1 - range_pct), 0), original_rate * (1 + range_pct), steps)
            chocs_values = np.linspace(max(1 - range_pct, 0), (1 + range_pct), steps)

            dv01_results = []
            convexity_results = []
            carry_results = []
            
            # Boucle explicite : On change le taux, on calcule, on stocke
            for r_simu in rate_values:
                self.rate = r_simu  # On modifie l'attribut de la classe temporairement
                
                # On appelle les méthodes (qui utiliseront le nouveau self.rate)
                dv01_results.append(self.dv01(list_risk_free_rates=[], fixed_rate=fixed_rate))
                convexity_results.append(self.convexity(list_risk_free_rates=[], fixed_rate=fixed_rate))
                carry_results.append(self.carry(list_risk_free_rates=[], fixed_rate=fixed_rate))
            data = {
                "choc sur le taux (%)": (chocs_values-1) * 100,
                "DV01": dv01_results,
                "Convexity": convexity_results,
                "Carry": carry_results
            }
        
        else:
            # On fait un déplacement parallèle de la courbe des taux
            # On aura donc une liste de listes
            chocs = np.linspace(max(1 - range_pct, 0), (1 + range_pct), steps)
            rate_values = [ [ri * choc for ri in list_risk_free_rates] for choc in chocs ]
            original_rate = self.rate

            dv01_results = []
            convexity_results = []
            carry_results = []

            for list_ri in rate_values:
                dv01_results.append(self.dv01(list_risk_free_rates=list_ri, fixed_rate=fixed_rate))
                convexity_results.append(self.convexity(list_risk_free_rates=list_ri, fixed_rate=fixed_rate))
                carry_results.append(self.carry(list_risk_free_rates=list_ri, fixed_rate=fixed_rate))
                
            data = {
                "choc sur le taux (%)": (chocs-1) *100,
                "DV01": dv01_results,
                "Convexity": convexity_results,
                "Carry": carry_results
            }
        
        # Restauration de la valeur originale du rate
        self.rate = original_rate
        
        data_df = pd.DataFrame(data)
        return data_df