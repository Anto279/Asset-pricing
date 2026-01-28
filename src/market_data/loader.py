import yfinance as yf
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt

class MarketDataLoader:
    
    def __init__(self):
        """Initialisation de la classe de chargement de données de marché."""
        
        # Mapping des Tickers FRED (St. Louis Fed) vers les maturités en années
        # DGS1MO = 1 Month Treasury Constant Maturity Rate
        self.fred_tickers = {
            'DGS1MO': 1/12,
            'DGS3MO': 3/12,
            'DGS6MO': 6/12,
            'DGS1': 1.0,
            'DGS2': 2.0,
            'DGS3': 3.0,
            'DGS5': 5.0,
            'DGS7': 7.0,
            'DGS10': 10.0,
            'DGS20': 20.0,
            'DGS30': 30.0
        }
        # Cache pour éviter de requêter l'API à chaque changement de slider
        self._yield_curve_cache = None

    def get_equity_spot(self, ticker: str) -> float:
        # Code existant inchangé pour la récupération du prix spot
        try:
            print(f"Téléchargement du spot pour {ticker}...")
            data = yf.download(ticker, period="5d", progress=False)
            
            if data.empty:
                raise ValueError("DataFrame vide : vérifiez le ticker ou la connexion.")
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if 'Close' in data.columns:
                last_price = data['Close'].iloc[-1]
                return float(last_price)
            else:
                raise ValueError("Colonne 'Close' manquante.")

        except Exception as e:
            raise ValueError(f"Erreur fatale lors de l'accès à Yahoo pour {ticker} : {e}")

    def get_data(self, ticker: str, start = None, end = None) -> pd.Series:
        """
        Récupère les N dernières valeurs de clôture pour l'actif spécifié.

        Args:
            ticker (str): Symbole Yahoo Finance.
            n_days (int): Nombre de jours de marché à retourner.

        Returns:
            pd.Series: Une série Pandas des prix de clôture.
        
        Raises:
            ValueError: Si le ticker est invalide ou si la colonne 'Close' est manquante.
        """
        
        # On demande une période plus longue (ex: 3 mois) pour s'assurer d'obtenir N jours
        tickerData = yf.Ticker(ticker)
        data = tickerData.history(start=start, end=end)

        
        if data.empty:
            raise ValueError("DataFrame vide : vérifiez le ticker ou la connexion.")
            
        return data  
    
    # TAUX SANS RISQUE (FRED) --- US Treaasury Yield Curve

    def get_us_treasury_curve(self):
        """
        Récupère la dernière courbe des taux US Treasury disponible sur FRED.
        Retourne une Pandas Series avec Index=Maturité (années) et Value=Taux (%).
        """
        if self._yield_curve_cache is not None:
             return self._yield_curve_cache

        try:
            # On cherche sur les 10 derniers jours pour être sûr de trouver une date de publication (jours ouvrés)
            start_date = datetime.datetime.now() - datetime.timedelta(days=10)
            end_date = datetime.datetime.now()
            
            # Récupération via pandas_datareader
            df = web.DataReader(list(self.fred_tickers.keys()), 'fred', start_date, end_date)
            
            # On prend la dernière ligne valide (la plus récente)
            latest_rates = df.iloc[-1].dropna()
            
            if latest_rates.empty:
                 raise ValueError("Aucune donnée récupérée de FRED.")

            # Construction de la courbe propre : Maturité -> Taux
            curve_data = {}
            for ticker, maturity in self.fred_tickers.items():
                if ticker in latest_rates:
                    curve_data[maturity] = latest_rates[ticker]
            
            # Conversion en Series triée par maturité
            curve_series = pd.Series(curve_data).sort_index()
            
            self._yield_curve_cache = curve_series
            return curve_series

        except Exception as e:
            print(f" Erreur lors de la récupération FRED : {e}")
            

    def get_risk_free_rate_from_curve(self, maturity: float) -> float:
        """
        Retourne le taux sans risque interpolé pour une maturité donnée.
        Retourne une valeur décimale (ex: 0.045 pour 4.5%).
        """
        curve = self.get_us_treasury_curve()
        
        maturities = curve.index.values
        rates = curve.values
        
        # Interpolation linéaire (numpy.interp)
        # Si la maturité est entre 3 et 5 ans, on calcule le taux intermédiaire
        interpolated_rate = np.interp(maturity, maturities, rates)
        
        return interpolated_rate / 100.0

    # Récupération des maturités d'options via yfinance associé à un ticker d'action

    def get_option_expirations(self, ticker: str) -> list:
        """Retourne la liste des dates d'expiration disponibles (ex: '2025-12-19')"""
        try:
            tk = yf.Ticker(ticker)
            if not tk.options:
                return []
            return list(tk.options)
        except Exception as e:
            print(f"Erreur lors de la récupération des expirations : {e}")
            return []
        
   

    def get_option_chain_data(self, ticker: str, expiration_date: str):
        """
        Récupère les données de marché pour les Calls et Puts à une date donnée.
        Renvoie un dictionnaire avec deux DataFrames : 'calls' et 'puts'.
        """
        try:
            tk = yf.Ticker(ticker)
            # option_chain renvoie un objet avec deux attributs : .calls et .puts
            chain = tk.option_chain(expiration_date)
            return {
                "calls": chain.calls,
                "puts": chain.puts
            }
        except Exception as e:
            raise ValueError(f"Impossible de récupérer la chaîne d'options pour {expiration_date} : {e}")
    
    def get_market_price_interpolated(self, ticker: str, expiration_date: str, strike: float, option_type="call") -> float:
        """
        Récupère le prix de marché interpolé pour n'importe quel strike,
        en se basant UNIQUEMENT sur le 'lastPrice' (dernier prix transigé).
        """
        # 1. Récupération des données
        data = self.get_option_chain_data(ticker, expiration_date)
        if data is None:
            return None
        
        # 2. Sélection du bon DataFrame (Calls ou Puts)
        df = data["calls"] if option_type == "call" else data["puts"]
        
        if df.empty:
            return None

        # 3. Nettoyage et Tri
        # On ne garde que les colonnes utiles : Strike et LastPrice
        # On supprime les lignes vides et on trie par strike (indispensable pour l'interpolation)
        df_clean = df[['strike', 'lastPrice']].dropna().sort_values('strike')
        
        strikes_disponibles = df_clean['strike'].values
        prix_last = df_clean['lastPrice'].values

        # 4. Interpolation Linéaire
        # np.interp va calculer le prix théorique 'lastPrice' pour votre strike
        interpolated_price = np.interp(strike, strikes_disponibles, prix_last)

        return float(interpolated_price)