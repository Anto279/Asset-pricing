import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import plotly.graph_objects as go

# Importation de vos modules
from datetime import datetime
from src.market_data.loader import MarketDataLoader 
from src.utils.parametres import Parametres
from src.instruments.bond import Bonds
from src.instruments.swap_taux import Swap_taux
from src.instruments.future import Future
from src.instruments.option_vanille import OptionVanille
from src.instruments.option_asiatique import OptionAsiatique 
from src.models.black_scholes import BlackScholesModel
from src.models.asian_model import AsianPricingModel 
from src.instruments.barrier_option import BarrierOption
from src.models.barrier_model import BarrierPricingModel
from src.models.monte_carlo import MonteCarloModel
from src.instruments.structured_product import StructuredProduct
from src.market_data.implied_volatility import ImpliedVolatility
from src.models.heston_model import HestonMonteCarloModel
from src.models.american_model import AmericanBinomialTreeModel, LongstaffSchwartzMonteCarloModel

import streamlit as st



def inject_css():
    st.markdown(
        """
        <style>
        /* --- global spacing tweaks --- */
        .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }

        /* --- nicer headers --- */
        h1, h2, h3 { letter-spacing: -0.2px; }

        /* --- “card” look when you use container(border=True) --- */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 16px;
        }

        /* --- tighten sidebar a bit --- */
        section[data-testid="stSidebar"] { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(
    page_title="Asset Pricing Project - ENSAI",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_css()

def section(title: str, caption: str | None = None, icon: str = ""):
    """Consistent section header + bordered container."""
    box = st.container(border=True)
    with box:
        st.subheader(f"{icon} {title}".strip())
        if caption:
            st.caption(caption)
    return box

def kpi_row(items):
    """
    items = [(label, value, delta(optional), help(optional)), ...]
    """
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        label = it[0]
        value = it[1]
        delta = it[2] if len(it) > 2 else None
        help_ = it[3] if len(it) > 3 else None
        c.metric(label, value, delta=delta, help=help_)

# OPTIONAL: cleaner separators everywhere
def divider():
    st.divider()


# --- Configuration de la Page ---
st.set_page_config(
    page_title="Asset Pricing Project - ENSAI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialisation des Outils ---
# On utilise le cache de Streamlit pour ne pas recharger la classe (et donc le cache FRED) à chaque re-run
@st.cache_resource
def get_loader():
    return MarketDataLoader()

DATA_LOADER = get_loader()

# =========================================================
# Helpers (cache + proxy) for option market data
# =========================================================

@st.cache_data(ttl=600, show_spinner=False)
def _spot_cached(ticker: str) -> float:
    return float(DATA_LOADER.get_equity_spot(ticker))

@st.cache_data(ttl=3600, show_spinner=False)
def _expirations_cached(ticker: str) -> list:
    return list(DATA_LOADER.get_option_expirations(ticker))

@st.cache_data(ttl=600, show_spinner=False)
def _chain_cached(ticker: str, expiration: str):
    return DATA_LOADER.get_option_chain_data(ticker, expiration)


# =========================================================
# Helpers (cache) for risk-free rate curve (FRED)
# =========================================================

@st.cache_data(ttl=24*3600, show_spinner=False)
def _yield_curve_cached():
    """Cache la courbe des taux US Treasury (FRED) pendant 24 h."""
    return DATA_LOADER.get_us_treasury_curve()


def rf_from_curve(maturity_years: float) -> float:
    """Taux sans risque interpolé (décimal) à partir de la courbe FRED."""
    curve = _yield_curve_cached()
    maturities = curve.index.to_numpy(dtype=float)
    rates = curve.to_numpy(dtype=float)
    return float(np.interp(float(maturity_years), maturities, rates)) / 100.0

# Yahoo Finance often doesn't provide option chains for indices (e.g. ^GSPC).
# We fall back to liquid ETF proxies when needed.
_OPTIONS_PROXY = {
    "^GSPC": "SPY",
    "^DJI": "DIA",
    "^IXIC": "QQQ",
}

def resolve_ticker_for_options(ticker: str):
    """Return (pricing_ticker, expirations, used_proxy)."""
    ticker = str(ticker).strip()
    expirations = _expirations_cached(ticker)
    if expirations:
        return ticker, expirations, False

    proxy = _OPTIONS_PROXY.get(ticker)
    if proxy:
        exp2 = _expirations_cached(proxy)
        return proxy, exp2, True

    return ticker, [], False

def nearest_expiration_index(expirations, target_T_years: float) -> int:
    """Pick the expiration closest to today + target_T_years."""
    if not expirations:
        return 0
    target = pd.Timestamp.today().normalize() + pd.Timedelta(days=int(max(float(target_T_years), 0.0) * 365))
    dts = pd.to_datetime(pd.Series(expirations), errors="coerce")
    if dts.isna().all():
        return 0
    return int((dts - target).abs().idxmin())


EQUITY_TICKERS = [
    # --- INDICES BOURSIERS MONDIAUX (Indices) ---
    "^GSPC (S&P 500)",          # USA - Large Cap
    "^DJI (Dow Jones)",         # USA - Industrial
    "^IXIC (NASDAQ Composite)", # USA - Tech
    "^FCHI (CAC 40)",           # France
    "^GDAXI (DAX 40)",          # Allemagne
    "^FTSE (FTSE 100)",         # Royaume-Uni
    "^N225 (Nikkei 225)",       # Japon
    "^HSI (Hang Seng)",         # Hong Kong
    "^VIX (Volatilité/Peur)",   # Indice de volatilité

    # --- TECHNOLOGIE & SEMICONDUCTEURS (USA) ---
    "AAPL (Apple)", 
    "MSFT (Microsoft)", 
    "GOOGL (Alphabet/Google)", 
    "AMZN (Amazon)", 
    "TSLA (Tesla)", 
    "NVDA (NVIDIA)",            # Leader IA/Puces
    "META (Meta/Facebook)", 
    "NFLX (Netflix)",
    "AMD (Advanced Micro Devices)",
    "INTC (Intel)",
    "CRM (Salesforce)",
    "ORCL (Oracle)",
    "ADBE (Adobe)",

    # --- FINANCE & BANQUES ---
    "JPM (JP Morgan Chase)", 
    "V (Visa)", 
    "MA (Mastercard)",
    "BAC (Bank of America)",
    "GS (Goldman Sachs)",
    "MS (Morgan Stanley)",
    "BLK (BlackRock)",
    "AXP (American Express)",

    # --- CONSOMMATION & RETAIL ---
    "WMT (Walmart)", 
    "KO (Coca-Cola)", 
    "PEP (PepsiCo)",
    "MCD (McDonald's)",
    "NKE (Nike)",
    "SBUX (Starbucks)",
    "COST (Costco)",
    "DIS (Disney)",
    "PG (Procter & Gamble)",

    # --- SANTÉ & PHARMA ---
    "JNJ (Johnson & Johnson)",
    "PFE (Pfizer)",
    "LLY (Eli Lilly)",          # Leader obésité/diabète
    "MRK (Merck)",
    "UNH (UnitedHealth)",

    # --- ÉNERGIE & INDUSTRIE ---
    "XOM (Exxon Mobil)",
    "CVX (Chevron)",
    "BA (Boeing)",
    "CAT (Caterpillar)",
    "GE (General Electric)",

    # --- ACTIONS FRANÇAISES (Suffixe .PA souvent requis) ---
    "MC.PA (LVMH)",             # Luxe
    "OR.PA (L'Oréal)",          # Cosmétique
    "TTE.PA (TotalEnergies)",   # Énergie
    "SAN.PA (Sanofi)",          # Santé
    "AIR.PA (Airbus)",          # Aéronautique
    "RMS.PA (Hermès)",          # Luxe
    "BNP.PA (BNP Paribas)"      # Banque
]

st.title("Projet de Pricing d'Actifs - ENSAI")
st.divider()

# --- 1. Barre Latérale (Sidebar) pour la Navigation ---
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Aller vers :", 
        ["Pricer", "Produits Structurés", "Calibration de la volatilité", "Surface de vol"],
        index=0
    )
    st.divider()
    st.markdown("### Info Projet")
    st.caption("Projet Asset Pricing ENSAI")
    st.caption("Données: Yahoo Finance & FRED")


def page_pricer():
    st.header("Pricer")

    # =================================================================
    # --- 2. Sélection de l'Instrument ---
    # =================================================================
    st.header("Sélection de l'Instrument")
        # Choix de l'instrument (Utilisation de st.selectbox)
    instrument_type = st.selectbox(
        "Quel instrument voulez-vous pricer ?",
        ["Bond (Obligation)", "Swap de Taux", "Future sur Actions", "Option Vanille sur Actions", "Option Asiatique", "Option Barrière"],
        index=3 # 'Bond (Obligation)' sélectionné par défaut
    )

    
    st.divider()
    # =================================================================
    # --- MINI-LEÇON : Aperçu rapide des instruments ---
    # =================================================================
    with st.expander("À quoi servent ces instruments ?", expanded=False):
        st.markdown(r"""
    Voici l’intuition générale, accompagnée d’un exemple illustratif, pour chaque type d’instrument financier.

    **1) Obligation (Bond)**  
    - **Principe** : l’investisseur prête un capital à un émetteur et reçoit des coupons périodiques ainsi que le remboursement du nominal à l’échéance.  
    - **Flux financiers** :
    $$
    \text{Valeur} = \sum_{t=1}^{T} C_t + N
    $$
    où $$C_t$$ désigne le coupon et $$N$$ le nominal.  
    - **Exemple** : nominal \(N = 1000\) €, coupon annuel de 2 % :
    $$
    C = 0.02 \times 1000 = 20 \text{ € par an}
    $$

    **2) Swap de taux d’intérêt**  
    - **Principe** : échange de flux d’intérêts entre un taux fixe et un taux variable, calculés sur un nominal notionnel \(N\).  
    - **Flux nets** :
    $$
    \text{Flux}_t = N \times (r_{\text{variable},t} - r_{\text{fixe}})
    $$
    - **Usage** : instrument principalement utilisé pour la couverture contre les variations des taux d’intérêt.

    **3) Contrat Forward / Future**  
    - **Principe** : engagement d’acheter ou de vendre un actif sous-jacent à une date future \(T\), à un prix fixé à l’avance \(F_0\).  
    - **Payoff à maturité (position longue)** :
    $$
    \text{Payoff} = S_T - F_0
    $$
    où $$S_T$$ est le prix du sous-jacent à maturité.

    **4) Option vanille (Call / Put)**  
    - **Call** : droit d’acheter le sous-jacent au prix d’exercice \(K\).  
    - **Put** : droit de vendre le sous-jacent au prix \(K\).  
    - **Payoffs à maturité** :
    $$
    \text{Call} = \max(S_T - K, 0)
    $$
    $$
    \text{Put} = \max(K - S_T, 0)
    $$

    **5) Option asiatique**  
    - **Principe** : le payoff dépend de la moyenne du prix du sous-jacent sur une période donnée, ce qui réduit l’impact des fluctuations extrêmes.  
    - **Payoff d’un call asiatique** :
    $$
    \text{Payoff} = \max\left( \frac{1}{n}\sum_{i=1}^{n} S_{t_i} - K, 0 \right)
    $$

    **6) Option barrière**  
    - **Principe** : option conditionnelle dont l’existence dépend du franchissement d’un niveau de barrière \(B\).  
    - **Exemple : Up-and-Out Call**  
    Si la barrière est atteinte avant l’échéance :
    $$
    \text{Payoff} = 0
    $$
    Sinon :
    $$
    \text{Payoff} = \max(S_T - K, 0)
    $$
    """)




    # =================================================================
    # --- 3. Entrée des Paramètres (Basée sur le type d'instrument) ---
    # =================================================================

    if instrument_type == "Bond (Obligation)":
        st.header(f"Paramètres pour : {instrument_type}")
        # --- PARTIE PÉDAGOGIQUE : BOND ---
        with st.expander("📘 Explication simple + exemple (Obligation)", expanded=False):
            st.markdown(r"""
        ### Intuition
        Une obligation correspond à un **prêt consenti par l’investisseur à un émetteur** (État ou entreprise).
        - L’investisseur perçoit des **coupons périodiques**.
        - À l’échéance, il récupère la **valeur nominale** de l’obligation.

        ### Pourquoi le prix varie-t-il ?
        Le prix d’une obligation aujourd’hui est égal à la **valeur actualisée des flux futurs** (coupons et nominal) :
        $$
        P_0 = \sum_{t=1}^{T} \frac{C}{(1+r)^t} + \frac{N}{(1+r)^T}
        $$
        où :
        - \(C\) est le coupon,
        - \(N\) la valeur nominale,
        - \(r\) le taux d’actualisation,
        - \(T\) la maturité.

        Conséquences économiques :
        - Si le **taux sans risque augmente**, les flux futurs sont davantage actualisés → **le prix de l’obligation diminue**.
        - Si l’émetteur présente un risque de crédit, on ajoute une **prime de risque (spread)** au taux d’actualisation :
        $$
        r = r_f + \text{spread}
        $$
        ce qui entraîne également une baisse du prix.

        ### Exemple illustratif
        Considérons une obligation avec :
        - Nominal \(N = 1000\) €,
        - Coupon annuel de 2 %, soit
        $$
        C = 0.02 \times 1000 = 20 \text{ €},
        $$
        - Maturité \(T = 5\) ans.

        Si le taux d’actualisation est $$r \approx 3\%$$, la valeur actuelle des flux est **inférieure à 1000 €**, car le coupon offert est relativement faible par rapport au taux du marché.
        """)


        # Utilisation de colonnes pour une meilleure disposition visuelle
        col1, col2, col3, col4, col5 = st.columns(5)

        # Paramètres Obligatoires (pour votre classe Parametres corrigée)
        with col1:
            # st.number_input pour des entrées numériques
            coupon_rate = st.number_input("Taux de Coupon (%)", min_value=0.0, value=2.0, step=0.1) / 100
        
        with col2:
            nominal_value = st.number_input("Valeur Nominale (€)", min_value=1.0, value=1000.0, step=100.0)
        
        with col3:
            maturity_years = st.number_input("Maturité (années)", min_value=1.0, value=5.0, step=1.0)

        # Gestion du Taux Sans Risque
        with col4:
            rate_source = st.selectbox("Taux sans Risque (r)", 
                ["Manuel", "Marché (FRED US Treasury)"],
                index=0
            )
            
            if rate_source == "Manuel":
                risk_free_rate = st.number_input("Taux d'Actualisation (r, %)", min_value=0.0, value=3.0, step=0.1) / 100
            else:
                # Récupération automatique et interpolation
                with st.spinner("Récupération de la courbe des taux..."):
                    risk_free_rate = rf_from_curve(maturity_years)
                st.success(f"Taux interpolé : **{risk_free_rate*100:.3f} %**")

        with col5:
            risk_profile = st.selectbox(
                "Catégorie de l'Obligation",
                ["Sans Risque / AAA", "Low Yield (Investment Grade)", "Medium Yield (BB)", "High Yield (Speculative/Junk)"]
            )
        
            # Mapping des spreads en fonction de la catégorie
            # Les spreads sont indicatifs (en points de pourcentage)
            spreads_map = {
                "Sans Risque / AAA": 0.00,        # 0 bps
                "Low Yield (Investment Grade)": 0.015, # 150 bps (ex: Entreprises solides)
                "Medium Yield (BB)": 0.035,       # 350 bps (ex: Entreprises plus risquées)
                "High Yield (Speculative/Junk)": 0.065 # 650 bps (ex: Entreprises en difficulté ou très endettées)
            }
            
            selected_spread = spreads_map[risk_profile]
            st.info(f"Prime de Risque ajoutée : **+{selected_spread*100:.2f} %**")
            
        # --- 4. Calcul et Affichage ---
        try:
            params = Parametres(
                coupon=nominal_value * coupon_rate,
                maturity=maturity_years,
                nominal=nominal_value,
                risk_free_rate=risk_free_rate + selected_spread
            )
            
            bond_to_price = Bonds(parameters=params)
            price_result = bond_to_price.pricing_bond()
            
            st.divider()
            st.subheader("Résultat du Pricing")
            
            col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
            with col_res1:
                st.metric("Prix de l'Obligation (PV)", f" {price_result:,.2f} €")
            with col_res2:
                st.metric("Pourcentage du Nominal", f" {(price_result/nominal_value)*100:,.2f} %")
            with col_res3:
                # Affichage de la courbe si FRED est sélectionné
                if rate_source == "Marché (FRED US Treasury)":
                    st.markdown("##### Courbe des taux US Treasury (Interpolation)")
                    curve_data = _yield_curve_cached()
            
                    # Création d'un graphique simple
                    chart_data = pd.DataFrame({"Taux (%)": curve_data})
                    st.line_chart(chart_data, height=200)
                    st.caption("Axe X : Maturité (Années) | Axe Y : Taux (%)")


        except Exception as e:
            st.error(f"Erreur lors du calcul : {e}")
            st.warning("Veuillez vérifier que les valeurs d'entrée sont valides et que les classes Parametres et Bonds sont correctement définies.")


    elif instrument_type == "Swap de Taux":
        st.header(f"Paramètres pour : {instrument_type}")
        # --- PARTIE PÉDAGOGIQUE : SWAP ---
        with st.expander("📘 Explication simple + exemple (Swap de taux d’intérêt)", expanded=False):
            st.markdown(r"""
                ### Intuition
                Un swap de taux d’intérêt est un **contrat d’échange de flux financiers** entre deux contreparties, calculés sur un nominal notionnel \(N\) :
                - **Jambe fixe** : paiement (ou réception) d’un taux fixe \(S\).
                - **Jambe variable** : réception (ou paiement) d’un taux variable, généralement indexé sur un taux de référence (Euribor, SOFR, etc.).

                ### Pourquoi est-ce utile ?
                - Une entreprise endettée à taux variable peut chercher à se **couvrir contre une hausse des taux** en payant un taux fixe et en recevant un taux variable.
                - Un investisseur peut utiliser un swap pour exprimer une **anticipation sur l’évolution future des taux d’intérêt**.

                ### Valorisation (principe)
                La valeur d’un swap est égale à la différence entre la valeur actuelle de la jambe variable et celle de la jambe fixe :
                $$
                \text{PV}_{\text{swap}} = \text{PV}_{\text{floating}} - \text{PV}_{\text{fixed}}
                $$

                - **Jambe fixe** :
                $$
                \text{PV}_{\text{fixed}} = \sum_{i=1}^{n} \frac{N \times S \times \Delta_i}{(1+r_i)^{t_i}}
                $$

                - **Jambe variable** (approximation standard) :
                $$
                \text{PV}_{\text{floating}} \approx N - \frac{N}{(1+r_T)^T}
                $$

                ### Exemple illustratif
                Considérons un swap avec :
                - Nominal \(N = 1\,000\,000\) €,
                - Maturité \(T = 5\) ans,
                - Fréquence de paiement semestrielle (\(n = 10\)),
                - Taux fixe \(S = 3\%\).

                Si les taux d’intérêt augmentent au cours du temps, la valeur de la **jambe variable** augmente, ce qui peut rendre la position **payeur fixe / receveur variable** favorable.

                ### Lecture des indicateurs affichés dans la page
                - **$$PV_{floating}$$** : valeur actuelle de la jambe variable.
                - **$$PV_{fixed}$$** : valeur actuelle de la jambe fixe.
                - **$$PV_{total}$$** :  
                $$
                \text{PV}_{\text{total}} = \text{PV}_{\text{floating}} - \text{PV}_{\text{fixed}}
                $$
                pour une position *payer fixe*.
                - **DV01** : sensibilité de la valeur du swap à une variation parallèle de **+1 point de base** des taux d’intérêt (mesure du risque de taux).
                """)


        # Entrée des paramètres pour le Swap (moins d'inputs nécessaires)
        col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
        
        with col_s1:
            swap_maturity = st.number_input("Maturité du Swap (années)", min_value=1.0, value=5.0, step=0.1)
        
        with col_s2:
            # La fréquence est essentielle pour l'annuity du Swap
            swap_frequency = st.number_input("Fréquence de Paiement (fois/an)", min_value=0, value=2, step=1) # 2x/an (semestriel) souvent par défaut

        with col_s3:
            swap_nominal = st.number_input("Nominal du Swap (€)", min_value=1.0, value= 100.0, step=100.0)

        with col_s4:
            rate_source = st.selectbox("Taux sans Risque (r)", 
                ["Manuel", "Marché (FRED US Treasury)"],
                index=0
            )

            list_risk_free_rates = []
            list_maturities = []

        with col_s5:
            if rate_source == "Manuel":
                swap_risk_free_rate = st.number_input("Taux d'Actualisation (r, %)", min_value=0.0, value=3.0, step=0.1) / 100
            else:
                # Récupération automatique et interpolation
                with st.spinner("Récupération de la courbe des taux..."):
                    swap_risk_free_rate = rf_from_curve(swap_maturity)
                    list_maturities = [(1/swap_frequency) * i for i in range(1, int(swap_maturity * swap_frequency) + 1)]
                    list_risk_free_rates = [rf_from_curve(T) for T in list_maturities]

        # --- Affichage des Données de Taux (Specifique Swap) ---
        if rate_source == "Marché (FRED US Treasury)" and list_risk_free_rates != []:
            st.divider()
            st.subheader("Analyse de la Structure par Terme")
            
            col_chart, col_df = st.columns([2, 1])
            
            with col_chart:
                st.markdown("###### Courbe des taux US Treasury (Globale)")
                st.line_chart(_yield_curve_cached(), height=300)
                
            with col_df:
                st.markdown("###### Taux utilisés pour le Pricing")
                # Création du DataFrame pour affichage
                df_rates = pd.DataFrame({
                    "Période": range(1, len(list_maturities) + 1),
                    "Maturité (Ans)": [f"{m:.2f}" for m in list_maturities],
                    "Taux Sans Risque (%)": [f"{r*100:.4f} %" for r in list_risk_free_rates]
                })
                st.dataframe(df_rates, hide_index=True, height=280)

        # --- 4. Calcul et Affichage ---
        try:
            # 4a. Instanciation de l'objet Parametres
            # On passe les arguments nécessaires au Swap et on laisse les autres par défaut
            swap_params = Parametres(
                maturity=swap_maturity,
                risk_free_rate=swap_risk_free_rate,
                frequency=swap_frequency,
                nominal=swap_nominal
            )
            
            # 4b. Instanciation du Swap
            swap_to_price = Swap_taux(parameters=swap_params)

            # 4c. Pricing (Taux de Swap "au pair")
            swap_pricing = swap_to_price.taux_swap(list_risk_free_rates=list_risk_free_rates)

            
            st.divider()
            st.subheader("Position payeur de swap — paiement du taux fixe et réception du taux variable")
            st.subheader("Résultat du Pricing")
            
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            # Il est intéressant de laisser le choix à l'utilisateur d'utiliser le taux fixe "au pair" ou un taux fixe manuel
            # Car dans la vraie vie, on doit souvent gérer des swap signés il y a queques temps avec un taux fixe connu.
            with col1: 
                fixed_manual_rate = st.selectbox("Taux Fixe du Swap S", 
                    ["Manuel", "Au pair (équitable)"],
                    index=1
                )

            with col2:
                if fixed_manual_rate == "Manuel":
                    user_fixed_rate = st.number_input("Taux Fixe Manuel S (%)", min_value=0.0, value=swap_pricing*100, step=0.01) / 100
                else:
                    user_fixed_rate = None # On utilisera le taux "au pair"
                
                swap_valuation_fixed = swap_to_price.valuation_swap_fixed(list_risk_free_rates, fixed_rate=user_fixed_rate)
                swap_valuation_floating = swap_to_price.valuation_swap_floating(list_risk_free_rates)
                valuation_swap = swap_to_price.valuation_swap(list_risk_free_rates, fixed_rate=user_fixed_rate)

            col_a, col_b, col_c, col_d = st.columns(4)

            with col_a:
                if fixed_manual_rate == "Manuel":
                    st.metric("Taux de Swap Manuel (S)", f"{user_fixed_rate * 100:,.3f} %")
                else:
                    st.metric("Taux de Swap (S) au pair (équitable)", f"{swap_pricing * 100:,.3f} %")
            with col_b:
                st.metric("Valuation floating : PV_floating", f" {swap_valuation_floating:,.2f} €")
            with col_c:
                st.metric("Valuation fixed : PV_fixed ", f" {swap_valuation_fixed:,.2f} €")
            with col_d:
                # Valuation doit être 0 si S est au pair
                st.metric("Valuation totale du Swap : PV_floating - PV_fixed", f" {valuation_swap:,.2f} €")

            st.divider()
            st.subheader("Sensibilités du Swap (Greeks)")

            col_dv01, col_dv01_graph = st.columns([1, 2])

            with col_dv01:
                dv01_value = swap_to_price.dv01(list_risk_free_rates, fixed_rate=user_fixed_rate)
                st.metric("DV01 (équivalent Delta)", f" {dv01_value:,.4f} €")
            with col_dv01_graph:
                # Graphique DV01
                st.subheader("DV01")
                dv01_data = swap_to_price.sensitivities_range(list_risk_free_rates, fixed_rate=user_fixed_rate, range_pct=5)
                st.line_chart(dv01_data, x="choc sur le taux (%)", y="DV01", height=250)

            col_convexity, col_convexity_graph = st.columns([1, 2])

            with col_convexity:
                convexity_value = swap_to_price.convexity(list_risk_free_rates, fixed_rate=user_fixed_rate)
                st.metric("Convexity (équivalent Gamma)", f" {convexity_value:,.4f} €")
            with col_convexity_graph:
                # Graphique Convexity
                st.subheader("Convexity")
                convexity_data = swap_to_price.sensitivities_range(list_risk_free_rates, fixed_rate=user_fixed_rate, range_pct=5)
                st.line_chart(convexity_data, x="choc sur le taux (%)", y="Convexity", height=250)
            
            col_carry, col_carry_graph = st.columns([1, 2])
            with col_carry:
                carry_value = swap_to_price.carry(list_risk_free_rates, fixed_rate=user_fixed_rate)
                st.metric("Carry (équivalent Theta)", f" {carry_value:,.4f} €")
            with col_carry_graph:
                # Graphique Carry
                st.subheader("Carry")
                carry_data = swap_to_price.sensitivities_range(list_risk_free_rates, fixed_rate=user_fixed_rate, range_pct=5)
                st.line_chart(carry_data, x="choc sur le taux (%)", y="Carry", height=250)
            
            col_graph1, col_graph2, col_graph3 = st.columns(3)

        except Exception as e:
            st.error(f"Erreur lors du calcul : {e}")
            st.warning("Veuillez vérifier que les valeurs d'entrée sont valides et que les classes Parametres et Swap_taux sont correctement définies.")


    elif instrument_type == "Future sur Actions":
        st.header(f"Paramètres pour : {instrument_type}")
        # --- PARTIE PÉDAGOGIQUE : FUTURE ---
        with st.expander("📘 Explication simple + exemple (Future / Forward)", expanded=False):
            st.markdown(r"""
        ### Intuition
        Un contrat **forward** (ou **future**) confère une **exposition directionnelle** au sous-jacent, sans paiement initial significatif.
        - **Position longue** : gain lorsque le prix du sous-jacent augmente.
        - **Position courte** : gain lorsque le prix du sous-jacent diminue.

        ### Pourquoi le prix forward diffère-t-il du prix spot ?
        En l’absence d’opportunités d’arbitrage, deux stratégies doivent avoir le même coût :
        - Acheter le sous-jacent aujourd’hui au prix spot $$S_0$$ et le financer au taux sans risque \(r\),
        - Attendre l’échéance \(T\) et acquérir le sous-jacent via un contrat forward.

        En présence d’un rendement continu \(q\) (dividendes, coupons ou convenience yield), le prix forward s’écrit :
        $$
        F_0 = S_0 \, e^{(r - q)T}
        $$

        ### Exemple illustratif
        Considérons les paramètres suivants :
        - Prix spot $$S_0 = 100$$,
        - Taux sans risque $$r = 3\%$$,
        - Rendement continu $$q = 1\%$$,
        - Maturité $$T = 1$$ an.

        Le prix forward est alors :
        $$
        F_0 = 100 \times e^{0.02} \approx 102.02
        $$
        """)


        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            # Sélection du Ticker
            selected_ticker_label = st.selectbox("Ticker de l'Actif Sous-jacent", EQUITY_TICKERS)
            
            # Extraction du ticker brut (ex: "AAPL")
            selected_ticker = selected_ticker_label.split(" ")[0]
            
            future_maturity = st.number_input("Maturité (années)", min_value=0.1, value=1.0, step=0.1)

        with col_f2:
            rate_source = st.selectbox("Taux sans Risque (r)", 
                ["Manuel", "Marché (FRED US Treasury)"],
                index=0
            )

            if rate_source == "Manuel":
                future_risk_free_rate = st.number_input("Taux d'Actualisation (r, %)", min_value=0.0, value=3.0, step=0.1) / 100
            else:
                # Récupération automatique et interpolation
                with st.spinner("Récupération de la courbe des taux..."):
                    future_risk_free_rate = rf_from_curve(future_maturity)
                st.success(f"Taux interpolé : **{future_risk_free_rate*100:.3f} %**")

        with col_f3:
            # Le dividende est la variable q dans la formule du Future
            dividend_yield = st.number_input("Rendement Dividende (q, %)", min_value=0.0, value=2.0, step=0.1) / 100


        # --- 4. Calcul et Affichage Future ---
        try:
            st.divider()
            
            # 4a. Récupération du Prix Spot
            # Supposons que MarketDataLoader a une méthode get_equity_spot(ticker)
            spot_price = DATA_LOADER.get_equity_spot(selected_ticker)
            
            
            # 4b. Instanciation de l'objet Parametres
            future_params = Parametres(
                maturity=future_maturity,
                risk_free_rate=future_risk_free_rate,
                dividend=dividend_yield # Ajout du rendement de dividende q
            )
            
            # 4c. Instanciation du Future
            future = Future(parameters=future_params, spot=spot_price)
            
            # 4d. Pricing (Formule F = S * exp((r-q)T)
            price_result_future = future.pricing_future()
            
            
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("Résultat du Pricing")
                st.metric("Prix du Contrat Future (F)", f" {price_result_future:,.2f} €")
                
            # recupération des données historiques pour affichage graphique
            data = DATA_LOADER.get_data(selected_ticker, end=pd.Timestamp.today(), start=pd.Timestamp.today() - pd.DateOffset(months=6))
                
            with col2:
                st.subheader(f"Prix de clôture des 6 derniers mois pour {selected_ticker}")
                st.line_chart(data['Close'], height=150) # Affichage du graphique des prix de clôture
                
        except Exception as e:
            st.error(f"Erreur lors du calcul du Future : {e}")
            st.warning(f"Impossible de pricer. Vérifiez l'implémentation de Future.pricing_future ou la fonction get_equity_spot. Erreur Python: {e}")

    elif instrument_type == "Option Vanille sur Actions":
        st.header(f"Paramètres pour : {instrument_type}")

        # =========================================================
        # Hypothèses & cadre théorique (présentation)
        # =========================================================
        with st.expander("Hypothèses des modèles (Black-Scholes / GBM MC / Heston)", expanded=False):
            st.markdown(r"""
        ### Hypothèses communes (cadre risque-neutre)
        - Valorisation sous la **probabilité risque-neutre** : drift théorique du sous-jacent \(r-q\).
        - **Absence d’arbitrage** (cadre de référence).
        - Option **européenne** : exercice uniquement à maturité.
        - Inputs $$r,\sigma ,q$$ **annualisés** et en **décimal** (ex : 20% → 0.20).
        """)

            st.markdown("### Black–Scholes (formule analytique)")
            st.markdown("- Sous-jacent modélisé par un **Mouvement Brownien Géométrique (GBM)** :")
            st.latex(r"dS_t = (r-q)\,S_t\,dt + \sigma S_t\,dW_t")
            st.markdown(r"- $$\sigma, r, q$$ sont **constants**.")
            st.markdown("- Marché sans friction (hypothèse standard), couverture continue (cadre idéal).")

            st.markdown("### Monte Carlo (GBM)")
            st.markdown("- Même dynamique GBM, mais estimation par simulation :")
            st.latex(r"\text{Prix} \approx e^{-rT}\,\frac{1}{N}\sum_{i=1}^{N}\text{Payoff}\!\left(S_T^{(i)}\right)")
            st.markdown(r"- **Erreur statistique** décroît en $$\frac{1}{sqrt(N)}$$.")
            st.markdown(r"- Si discrétisation temporelle : **erreur de discrétisation** (dépend du nombre de steps).")

            st.markdown("### Heston (volatilité/variance stochastique)")
            st.markdown("- Variance stochastique (type CIR) :")
            st.latex(r"dS_t = (r-q)\,S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{S}")
            st.latex(r"dv_t = \kappa(\theta - v_t)\,dt + x_i\sqrt{v_t}\,dW_t^{v}, \qquad \mathrm{corr}(dW^{S}, dW^{v})=\rho")
            st.markdown(r"- $$\kappa$$ : retour à la moyenne, $$\theta$$ : variance long-terme, $$x_i$$ : vol-of-vol, $$\rho$$ : corrélation.")
            st.markdown("- Condition de Feller (positivité “forte” de la variance) :")
            st.latex(r"2\kappa\theta \ge x_i^{2}")
            st.markdown(r"- Si violée, $$v_t$$ peut s’approcher plus souvent de 0 (selon le schéma numérique, ça peut aussi introduire un biais).")

            st.markdown("### Remarque sur le dividende (q)")
            st.markdown(r"""
        - $$q$$ représente un **rendement de dividende continu**.
        """)


        # =========================================================
        # Partie pédagogique (option vanille)
        # =========================================================
        with st.expander("📘 Explication simple + exemples (Option vanille)", expanded=False):
            st.markdown(r"""
        ### Intuition
        Une option est un **instrument dérivé conférant un droit, et non une obligation**, portant sur un actif sous-jacent.

        - **Option Call** : droit d’acheter le sous-jacent au prix d’exercice \(K\).
        $$
        \text{Payoff}_{\text{Call}} = \max(S_T - K, 0)
        $$

        - **Option Put** : droit de vendre le sous-jacent au prix d’exercice \(K\).
        $$
        \text{Payoff}_{\text{Put}} = \max(K - S_T, 0)
        $$

        ### Lecture synthétique des grecques
        Les **grecques** mesurent la sensibilité du prix de l’option aux différents paramètres du modèle :
        - **Delta $$\Delta$$** : sensibilité du prix de l’option à une variation du prix du sous-jacent.
        - **Gamma $$\Gamma$$** : variation du delta par rapport au sous-jacent (mesure de la convexité).
        - **Vega $$\nu$$** : sensibilité du prix à une variation de la volatilité.
        - **Theta $$\Theta$$** : impact de l’écoulement du temps sur la valeur de l’option.
        - **Rho $$\rho$$** : sensibilité du prix de l’option à une variation du taux d’intérêt.
        """)

        # =========================================================
        # Helpers American (local, no dependency)
        # =========================================================
        def _intrinsic(S, K, opt_type: str):
            S = np.asarray(S, dtype=float)
            if str(opt_type).lower() == "call":
                return np.maximum(S - K, 0.0)
            return np.maximum(K - S, 0.0)

        def price_american_crr(S0, K, T, r, q, sigma, steps: int, opt_type: str):
            """CRR binomial with early exercise (American)."""
            steps = int(steps)
            if steps <= 0:
                return float(_intrinsic(S0, K, opt_type))

            sigma = float(max(sigma, 1e-12))
            dt = float(T) / steps
            disc = np.exp(-float(r) * dt)

            u = np.exp(sigma * np.sqrt(dt))
            d = 1.0 / u

            # Risk-neutral p with dividend yield q
            a = np.exp((float(r) - float(q)) * dt)
            denom = (u - d)
            if abs(denom) < 1e-16:
                # degenerate case ~ zero vol
                ST = S0 * (a ** steps)
                return float(np.exp(-r * T) * _intrinsic(ST, K, opt_type))

            p = (a - d) / denom
            p = float(np.clip(p, 0.0, 1.0))

            # Terminal stock prices
            j = np.arange(steps + 1)
            ST = float(S0) * (u ** j) * (d ** (steps - j))

            # Terminal option values
            V = _intrinsic(ST, float(K), opt_type).astype(float)

            # Backward induction with early exercise
            for i in range(steps - 1, -1, -1):
                j = np.arange(i + 1)
                S_ij = float(S0) * (u ** j) * (d ** (i - j))
                continuation = disc * (p * V[1:i + 2] + (1.0 - p) * V[0:i + 1])
                exercise = _intrinsic(S_ij, float(K), opt_type)
                V = np.maximum(exercise, continuation)

            return float(V[0])

        def price_american_lsm(
            S0, K, T, r, q, sigma,
            n_paths: int, steps: int,
            opt_type: str,
            degree: int = 2,
            seed: int = 123,
            antithetic: bool = True
        ):
            """
            Longstaff-Schwartz (LSM) under GBM with continuous dividend yield q.
            Returns: (price, paths_matrix)  paths_matrix shape (n_paths, steps+1)
            """
            def _intrinsic(S, K, opt_type: str):
                S = np.asarray(S, dtype=float)
                if str(opt_type).lower() == "call":
                    return np.maximum(S - K, 0.0)
                return np.maximum(K - S, 0.0)

            n_paths = int(n_paths)
            steps = int(steps)
            degree = int(degree)

            if steps <= 0 or n_paths <= 0 or T <= 0:
                return float(_intrinsic(S0, K, opt_type)), None

            sigma = float(max(sigma, 1e-12))
            dt = float(T) / steps
            disc = np.exp(-float(r) * dt)

            rng = np.random.default_rng(int(seed))

            # Z with optional antithetic variance reduction
            if antithetic:
                half = (n_paths + 1) // 2
                Z_half = rng.standard_normal(size=(half, steps))
                Z = np.vstack([Z_half, -Z_half])[:n_paths, :]
            else:
                Z = rng.standard_normal(size=(n_paths, steps))

            S = np.zeros((n_paths, steps + 1), dtype=float)
            S[:, 0] = float(S0)

            drift = (float(r) - float(q) - 0.5 * sigma * sigma) * dt
            volterm = sigma * np.sqrt(dt)

            for t in range(steps):
                S[:, t + 1] = S[:, t] * np.exp(drift + volterm * Z[:, t])

            # cashflow at maturity (time = steps)
            cf = _intrinsic(S[:, -1], float(K), opt_type).astype(float)

            # Backward induction from steps-1 down to 1
            for t in range(steps - 1, 0, -1):
                cf *= disc  # discount one step back (to time t)

                St = S[:, t]
                ex = _intrinsic(St, float(K), opt_type)

                itm = ex > 0.0
                if not np.any(itm):
                    continue

                # IMPORTANT: scale the state variable for numerical stability
                x = (St[itm] / float(K))
                y = cf[itm]  # continuation value already discounted to time t

                # Polynomial basis: [1, x, x^2, ..., x^degree]
                X = np.vstack([x ** k for k in range(degree + 1)]).T
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                cont = X @ beta

                # Exercise decision (no "alive" mask here)
                ex_now = ex[itm] > cont
                idx_itm = np.where(itm)[0]
                idx_ex = idx_itm[ex_now]

                cf[idx_ex] = ex[idx_ex]  # set cashflow to immediate exercise payoff at time t

            # Discount from time 1 to time 0
            cf *= disc
            price = float(np.mean(cf))

            # Allow immediate exercise at t=0
            price = max(price, float(_intrinsic(float(S0), float(K), opt_type)))

            return price, S


        st.divider()

        # =========================================================
        # Style d'exercice + méthode de pricing
        # =========================================================
        exercise_style = st.selectbox("Style d'exercice", ["European", "American"], index=0)

        if exercise_style == "European":
            pricing_method = st.radio(
                "Méthode de Valorisation",
                [
                    "Formule de Black-Scholes (Analytique)",
                    "Monte Carlo (Simulation - GBM)",
                    "Heston (Monte Carlo - Volatilité Stochastique)"
                ],
                horizontal=True
            )
        else:
            pricing_method = st.radio(
                "Méthode de Valorisation (American)",
                [
                    "Binomial Tree (CRR - Early Exercise)",
                    "Monte Carlo (Longstaff-Schwartz - LSM)"
                ],
                horizontal=True
            )

        st.divider()

        # =========================================================
        # 1) Inputs communs
        # =========================================================
        col_o1, col_o2, col_o3, col_o4, col_o5, col_o6 = st.columns(6)

        with col_o1:
            selected_ticker_label = st.selectbox("Ticker de l'Actif Sous-jacent", EQUITY_TICKERS)
            selected_ticker = selected_ticker_label.split(" ")[0]
            option_maturity = st.number_input("Maturité (années)", min_value=0.01, value=1.0, step=0.1)
            st.caption("T en années (ex : 0.5 = 6 mois).")

        with col_o2:
            spot_temp = 100.0
            try:
                spot_temp = float(DATA_LOADER.get_equity_spot(selected_ticker))
            except Exception:
                pass
            option_strike = st.number_input("Prix d'Exercice (K, €)", min_value=0.1, value=float(spot_temp), step=5.0)
            st.caption("Strike \(K\).")

        with col_o3:
            vol_source = st.selectbox("Source de volatilité (σ)", ["Manuel", "Marché (IV Yahoo)"], index=0)

            if vol_source == "Manuel":
                option_volatility = st.number_input("Volatilité (σ, %)", min_value=0.0, value=20.0, step=0.1) / 100
                st.caption("σ annualisée, saisie en % puis convertie en décimal.")
            else:
                option_volatility = None
                st.caption("σ récupérée depuis les options cotées (volatilité implicite).")

        with col_o4:
            option_type = st.selectbox("Type d'Option", ["call", "put"])
            st.caption("Option vanille (style sélectionné ci-dessus).")

        with col_o5:
            rate_source = st.selectbox("Risk Free Rate Source", ["Manuel", "Marché (FRED US Treasury)"], index=0)
            if rate_source == "Manuel":
                option_risk_free_rate = st.number_input("Taux d'Actualisation (r, %)", min_value=0.0, value=3.0, step=0.1) / 100
                st.caption("r annualisé, saisi en % puis converti en décimal.")
            else:
                with st.spinner("Récupération de la courbe des taux..."):
                    option_risk_free_rate = float(rf_from_curve(option_maturity))
                st.success(f"Taux interpolé : **{option_risk_free_rate*100:.3f} %**")
                st.caption("r obtenu via interpolation sur une courbe de taux.")

        with col_o6:
            dividend_yield = st.number_input("Rendement Dividende (q, %)", min_value=0.0, value=0.0, step=0.1) / 100
            st.caption("q = dividend yield continu (mettre 0 si non utilisé).")

        # =========================================================
        # Volatilité : saisie manuelle ou volatilité implicite (Yahoo)
        # =========================================================
        spot_ticker_for_pricing = selected_ticker
        K_user = float(option_strike)
        option_strike_used = float(option_strike)
        used_proxy = False
        option_expiration = None

        if vol_source != "Manuel":
            with st.expander("⚙️ Volatilité implicite de marché (Yahoo Finance)", expanded=True):
                pricing_ticker, expirations, used_proxy = resolve_ticker_for_options(selected_ticker)
                spot_ticker_for_pricing = pricing_ticker if used_proxy else selected_ticker

                if used_proxy:
                    st.info(
                        f"Pas de chaîne d'options disponible pour **{selected_ticker}** sur Yahoo "
                        f"→ utilisation du proxy **{pricing_ticker}**."
                    )

                if not expirations:
                    st.error("Aucune échéance d'options trouvée sur Yahoo Finance (ni pour ce ticker, ni pour un proxy).")
                    # Fallback: keep original underlying/strike if market IV is unavailable
                    spot_ticker_for_pricing = selected_ticker
                    used_proxy = False
                else:
                    default_idx = nearest_expiration_index(expirations, option_maturity)
                    option_expiration = st.selectbox("Expiration (pour IV)", expirations, index=default_idx)

                    iv_mode = st.radio(
                        "IV utilisée",
                        ["impliedVolatility (Yahoo)", "Recalculée via Black-Scholes + prix marché"],
                        index=0,
                        horizontal=True
                    )
                    price_mode = st.radio(
                        "Prix marché",
                        ["mid (bid/ask)", "lastPrice"],
                        index=0,
                        horizontal=True
                    )

                    # Spots (index vs proxy) + conversion du strike si proxy
                    spot_index = _spot_cached(selected_ticker)
                    spot_proxy = _spot_cached(pricing_ticker)

                    if used_proxy:
                        moneyness = K_user / max(spot_index, 1e-12)
                        option_strike_used = moneyness * spot_proxy
                        st.caption(
                            f"Conversion strike (moneyness conservée) : "
                            f"K_index={K_user:.2f} / S_index={spot_index:.2f} = m={moneyness:.4f} ⇒ "
                            f"K_proxy={option_strike_used:.2f} (S_proxy={spot_proxy:.2f})"
                        )
                    else:
                        option_strike_used = K_user

                    # Maturité à partir de l'expiration (cohérent avec IV)
                    try:
                        exp_date = datetime.strptime(option_expiration, "%Y-%m-%d")
                        days_to_mat = max((exp_date - datetime.now()).days, 1)
                        T_market = days_to_mat / 365.0
                        option_maturity = float(T_market)
                        st.caption(f"Maturité utilisée (d'après expiration) : **T={option_maturity:.4f} ans** (~{days_to_mat} jours).")

                        if rate_source != "Manuel":
                            option_risk_free_rate = float(rf_from_curve(option_maturity))
                            st.caption(f"Taux FRED recalculé pour ce T : **{option_risk_free_rate*100:.3f}%**")
                    except Exception as e:
                        st.warning(f"Impossible de convertir l'expiration en maturité : {e}")

                    # Récupération chaîne + IV
                    try:
                        chain = _chain_cached(pricing_ticker, option_expiration)
                        df_opt = chain["calls"] if option_type == "call" else chain["puts"]

                        if df_opt is None or df_opt.empty:
                            st.error("Chaîne d'options vide (données indisponibles / faible liquidité).")
                        else:
                            df_opt = df_opt.copy()
                            df_opt["abs_diff"] = (df_opt["strike"] - float(option_strike_used)).abs()
                            row = df_opt.loc[df_opt["abs_diff"].idxmin()]

                            strike_near = float(row.get("strike", option_strike_used))
                            last_px = float(row.get("lastPrice", np.nan))
                            bid = float(row.get("bid", np.nan))
                            ask = float(row.get("ask", np.nan))
                            mid_px = (bid + ask) / 2.0 if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0 else np.nan
                            iv_yahoo = float(row.get("impliedVolatility", np.nan))

                            market_price = mid_px if (price_mode.startswith("mid") and np.isfinite(mid_px)) else last_px

                            c1, c2, c3 = st.columns(3)
                            c1.metric("Strike (proche)", f"{strike_near:.2f}")
                            c2.metric("Prix marché", f"{market_price:.4f}" if np.isfinite(market_price) else "n/a")
                            c3.metric("IV Yahoo", f"{iv_yahoo*100:.2f}%" if np.isfinite(iv_yahoo) else "n/a")

                            sigma_market = None

                            # 1) Take Yahoo's impliedVolatility if available
                            if iv_mode.startswith("impliedVolatility") and np.isfinite(iv_yahoo) and iv_yahoo > 0:
                                sigma_market = float(iv_yahoo)

                            # 2) Or recompute IV from a market price (last / mid) via BS
                            else:
                                if np.isfinite(market_price) and market_price > 0:
                                    bs_model = BlackScholesModel()
                                    params_iv = Parametres(
                                        maturity=float(option_maturity),
                                        risk_free_rate=float(option_risk_free_rate),
                                        strike=float(option_strike_used),
                                        volatility=0.2,
                                        dividend=float(dividend_yield),
                                    )
                                    opt_iv = OptionVanille(
                                        parameters=params_iv,
                                        spot=float(spot_proxy if used_proxy else spot_index),
                                        option=option_type
                                    )

                                    iv_calc = ImpliedVolatility(float(market_price), opt_iv, bs_model)
                                    sigma_market = iv_calc.newton_raphson(initial_guess=0.2)
                                    if sigma_market is None:
                                        sigma_market = iv_calc.dichotomie()

                            if sigma_market is None:
                                st.error("Impossible de récupérer/calculer une IV fiable → fallback sur σ=20%.")
                                option_volatility = 0.20
                            else:
                                option_volatility = float(sigma_market)
                                st.success(f"Volatilité utilisée : **{option_volatility*100:.2f}%**")

                            # Pour le pricing: utiliser le strike converti si proxy actif
                            option_strike = float(option_strike_used)

                    except Exception as e:
                        st.error(f"Erreur lors de la récupération IV : {e}")
                        option_volatility = 0.20

        # Ensure sigma is always defined (avoid crashes in Heston defaults etc.)
        if option_volatility is None:
            option_volatility = 0.20

        # =========================================================
        # 2) Params spécifiques (GBM MC / Heston / American)
        # =========================================================
        n_simulations_gbm = None
        n_paths_visu = None
        n_steps_visu = None

        # American params
        steps_tree = None
        n_paths_lsm = None
        steps_lsm = None
        degree_lsm = None
        antithetic_lsm = None

        if exercise_style == "European":
            if "Monte Carlo" in pricing_method and "GBM" in pricing_method:
                st.markdown("###### Paramètres de Simulation (GBM)")
                st.caption(r"N ↑ → variance de l’estimateur ↓ (≈ $\frac{1}{\sqrt{N}}$)")
                mc_col1, mc_col2, mc_col3 = st.columns(3)
                with mc_col1:
                    n_simulations_gbm = st.slider("Nombre de simulations (N)", 1000, 100000, 200000, 1000)
                with mc_col2:
                    n_paths_visu = st.slider("Nombre de trajectoires affichées", 10, 300, 50)
                with mc_col3:
                    n_steps_visu = st.slider("Pas de temps (trajectoires)", 10, 252, 80)

        if exercise_style == "American":
            if "Binomial Tree" in pricing_method:
                st.markdown("###### Paramètres Binomial Tree (CRR)")
                steps_tree = st.slider("Steps (Tree)", 50, 3000, 800, 50)
            else:
                st.markdown("###### Paramètres Longstaff-Schwartz (LSM)")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    n_paths_lsm = st.slider("Nombre de paths (LSM)", 5000, 200000, 40000, 5000)
                with c2:
                    steps_lsm = st.slider("Steps (LSM)", 25, 500, 50, 25)
                with c3:
                    degree_lsm = st.selectbox("Degré polynôme (LSM)", [1, 2, 3], index=1)
                with c4:
                    antithetic_lsm = st.checkbox("Antithetic (LSM)", value=True)

        # Heston params (unchanged)
        heston_params = None
        h_n_paths = None
        h_steps = None
        h_seed = None
        h_antithetic = None
        h_show_paths = None

        if exercise_style == "European" and "Heston" in pricing_method:
            st.markdown("###### Paramètres Heston (variance stochastique)")
            h1, h2, h3, h4, h5 = st.columns(5)

            vol2_default = float(option_volatility**2)

            with h1:
                v0 = st.number_input("$$V0$$ (variance initial)", min_value=0.0, value=vol2_default, step=0.001, format="%.4f")
            with h2:
                kappa = st.number_input("$$\kappa$$", min_value=0.0001, value=1.5, step=0.1, format="%.4f")
            with h3:
                theta = st.number_input(r"$\theta$ (var long terme)", min_value=0.0, value=vol2_default, step=0.001, format="%.4f")
            with h4:
                xi = st.number_input("$$x_i$$ (vol-of-vol)", min_value=0.0, value=0.5, step=0.05, format="%.4f")
            with h5:
                rho = st.slider(r"$$\rho$$", min_value=-0.999, max_value=0.999, value=-0.7, step=0.01)


            st.markdown("###### Réglages Monte Carlo (Heston)")
            hh1, hh2, hh3, hh4 = st.columns(4)
            with hh1:
                h_n_paths = st.slider("Nombre de paths (Heston)", 2000, 500000, 200000, 2000)
            with hh2:
                h_steps = st.slider("Steps (Heston)", 50, 1000, 252, 10)
            with hh3:
                h_seed = st.number_input("Seed", min_value=0, value=123, step=1)
            with hh4:
                h_antithetic = st.checkbox("Antithetic", value=True)

            h_show_paths = st.checkbox("Afficher trajectoires Heston (S & v)", value=True)

            heston_params = dict(
                v0=float(v0),
                kappa=float(kappa),
                theta=float(theta),
                xi=float(xi),
                rho=float(rho),
            )

        # =========================================================
        # 3) Pricing
        # =========================================================
        try:
            st.divider()

            # Spot actuel
            spot_price = float(_spot_cached(spot_ticker_for_pricing))

            # Objet OptionVanille (instrument) + inject dividend into Parametres
            option_params = Parametres(
                maturity=option_maturity,
                risk_free_rate=option_risk_free_rate,
                strike=option_strike,
                volatility=option_volatility,
                dividend=dividend_yield,
            )
            option = OptionVanille(parameters=option_params, spot=spot_price, option=option_type)

            price_result_option = None
            heston_out = None

            # Keep references for later sections
            mc_model = None
            lsm_paths_mat = None

            if exercise_style == "European":
                if "Black-Scholes" in pricing_method:
                    bs_model = BlackScholesModel()
                    price_result_option = bs_model.price_option(option)

                elif "Monte Carlo" in pricing_method and "GBM" in pricing_method:
                    if n_simulations_gbm is None:
                        n_simulations_gbm = 1000000
                    mc_model = MonteCarloModel(num_simulations=n_simulations_gbm)
                    with st.spinner("Simulation Monte Carlo (GBM) en cours..."):
                        price_result_option = mc_model.price_option(option)

                elif "Heston" in pricing_method:
                    heston_model = HestonMonteCarloModel(chunk_size=20000)
                    st.markdown("### Pourquoi Heston ne donne pas le même prix que Black–Scholes (même en mettant $$\sigma = 20\%$$)")
                    st.markdown("""
                    Black–Scholes et Heston ne “voient” pas la même chose :

                    - **BS** utilise une volatilité **constante** (un seul paramètre).
                    - **Heston** fait évoluer la variance **au cours du temps** : le prix dépend de la distribution de la variance intégrée, pas seulement de sa moyenne.
                    """)

                    # Formule clé : variance intégrée
                    st.latex(r"\int_0^T v_t\,dt")

                    st.markdown(r"""
                    Conséquences pratiques :
                    - Même en prenant $$v_0=\sigma^2$$ et $$\theta=\sigma^2$$, tu peux avoir un prix différent si la variance est aléatoire (vol-of-vol).
                    - La corrélation change la forme de la distribution (skew) : avec $$\rho<0$$, les calls ATM peuvent être **moins chers** que BS à vol “moyenne” égale.
                    - Si la **condition de Feller** est violée, certains schémas de discrétisation peuvent garder $$v_t$$ trop souvent proche de 0 → vol effective plus faible → prix plus bas.
                    - Si les paramètres Heston ne sont **pas calibrés** sur une surface de volatilité implicite, tu prices juste “un autre monde” que celui résumé par « $$\sigma = 20\%$$ ».
                    """)

                    # Relations utiles en LaTeX
                    st.markdown("Relations (rappel) :")
                    st.latex(r"v_0=\sigma^2,\quad \theta=\sigma^2")
                    st.latex(r"2\kappa\theta \ge x_i^2 \quad \text{(condition de Feller)}")

                    st.markdown(r"""**Vérification (Heston ≈ Black–Scholes)** : en imposant une variance quasi déterministe (par exemple $x_i \approx 0$) et en fixant $v_0=\theta=\sigma^2$, le modèle de Heston **se réduit asymptotiquement** à Black–Scholes ; les prix obtenus doivent alors être **très proches**, à l’erreur Monte Carlo près.""")

                    st.latex(r"x_i \approx 0,\quad v_0=\theta=\sigma^2")


                    with st.spinner("Simulation Monte Carlo (Heston) en cours..."):
                        heston_out = heston_model.price_european_mc(
                            S0=spot_price,
                            K=float(option_strike),
                            T=float(option_maturity),
                            r=float(option_risk_free_rate),
                            q=float(dividend_yield),
                            v0=heston_params["v0"],
                            kappa=heston_params["kappa"],
                            theta=heston_params["theta"],
                            xi=heston_params["xi"],
                            rho=heston_params["rho"],
                            option_type=str(option_type).lower(),
                            n_paths=int(h_n_paths),
                            steps=int(h_steps),
                            seed=int(h_seed),
                            antithetic=bool(h_antithetic),
                            full_truncation=True,
                        )
                    price_result_option = float(heston_out["price"])

                else:
                    raise ValueError("Méthode de pricing inconnue (European).")

            else:
                # ===== American pricing =====
                if "Binomial Tree" in pricing_method:
                    if steps_tree is None:
                        steps_tree = 800
                    with st.spinner("Binomial Tree (CRR) en cours..."):
                        price_result_option = price_american_crr(
                            S0=float(spot_price),
                            K=float(option_strike),
                            T=float(option_maturity),
                            r=float(option_risk_free_rate),
                            q=float(dividend_yield),
                            sigma=float(option_volatility),
                            steps=int(steps_tree),
                            opt_type=str(option_type).lower()
                        )

                elif "Longstaff-Schwartz" in pricing_method:
                    if n_paths_lsm is None:
                        n_paths_lsm = 40000
                    if steps_lsm is None:
                        steps_lsm = 100
                    if degree_lsm is None:
                        degree_lsm = 2
                    if antithetic_lsm is None:
                        antithetic_lsm = True

                    with st.spinner("Monte Carlo (LSM) en cours..."):
                        price_result_option, lsm_paths_mat = price_american_lsm(
                            S0=float(spot_price),
                            K=float(option_strike),
                            T=float(option_maturity),
                            r=float(option_risk_free_rate),
                            q=float(dividend_yield),
                            sigma=float(option_volatility),
                            n_paths=int(n_paths_lsm),
                            steps=int(steps_lsm),
                            opt_type=str(option_type).lower(),
                            degree=int(degree_lsm),
                            seed=123,
                            antithetic=bool(antithetic_lsm),
                        )
                else:
                    raise ValueError("Méthode de pricing inconnue (American).")

            # =========================================================
            # 4) Affichage (prix + CI si Heston)
            # =========================================================
            col_res1, col_res2 = st.columns([2, 1])

            with col_res1:
                st.subheader(f"Historique : {selected_ticker}")
                data_option = DATA_LOADER.get_data(
                    selected_ticker,
                    end=pd.Timestamp.today(),
                    start=pd.Timestamp.today() - pd.DateOffset(years=2)
                )
                st.line_chart(data_option["Close"], height=200)

            with col_res2:
                st.markdown(f"""
                    <div style="text-align: center; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                        <h4 style="margin:0;">Prix de l'Option</h4>
                        <h1 style="color: #4CAF50;">{price_result_option:,.4f} €</h1>
                        <p>Spot: {spot_price:.2f} € | Strike: {float(option_strike):.2f} €</p>
                        <p>Style: {exercise_style} | Méthode: {pricing_method}</p>
                    </div>
                """, unsafe_allow_html=True)

                if heston_out is not None:
                    st.caption(f"Std error: {heston_out['std_error']:.6f}")
                    st.caption(f"IC 95%: [{heston_out['ci_95'][0]:.6f} ; {heston_out['ci_95'][1]:.6f}]")
                    st.caption(f"Paths: {heston_out['n_paths']} | Steps: {heston_out['steps']}")

            st.divider()

            # =========================================================
            # 5) Trajectoires (GBM / Heston)
            # =========================================================
            if exercise_style == "European" and ("Monte Carlo" in pricing_method and "GBM" in pricing_method):
                st.subheader("Visualisation des Trajectoires (GBM)")
                st.caption(f"{n_paths_visu} trajectoires sur {n_steps_visu} steps.")
                paths = mc_model.simulate_price_paths( spot_price, option_risk_free_rate, option_volatility, option_maturity, n_steps_visu, q=dividend_yield, n_paths=n_paths_visu, seed = 123)

                subset_paths = paths[:n_paths_visu, :].T
                df_paths = pd.DataFrame(subset_paths, columns=[f"Sim_{i}" for i in range(n_paths_visu)])
                time_index = np.linspace(0, option_maturity, n_steps_visu + 1)
                df_paths["Time"] = time_index
                df_melted = df_paths.melt("Time", var_name="Simulation", value_name="Prix")

                chart_mc = alt.Chart(df_melted).mark_line(opacity=0.6, strokeWidth=1.5).encode(
                    x=alt.X('Time', title='Années'),
                    y=alt.Y('Prix', title='Prix du Sous-Jacent', scale=alt.Scale(zero=False)),
                    color=alt.Color('Simulation:N', legend=None),
                    tooltip=['Time', 'Prix', 'Simulation']
                ).properties(height=400, title="Simulation GBM").interactive()

                rule = alt.Chart(pd.DataFrame({'y': [float(option_strike)]})).mark_rule(color='black', strokeWidth=2, strokeDash=[5,5]).encode(y='y')
                st.altair_chart(chart_mc + rule, use_container_width=True)

            if exercise_style == "American" and ("Longstaff-Schwartz" in pricing_method) and (lsm_paths_mat is not None):
                st.subheader("Visualisation des Trajectoires (GBM - LSM)")
                st.caption("Trajectoires simulées utilisées par LSM (sous GBM).")
                n_paths_show = st.slider("Trajectoires affichées (LSM)", 10, 200, 40)
                steps_show = lsm_paths_mat.shape[1] - 1
                subset = lsm_paths_mat[:n_paths_show, :].T
                df_paths = pd.DataFrame(subset, columns=[f"Path_{i}" for i in range(n_paths_show)])
                df_paths["Time"] = np.linspace(0, option_maturity, steps_show + 1)
                df_melted = df_paths.melt("Time", var_name="Path", value_name="Prix")
                chart_lsm = alt.Chart(df_melted).mark_line(opacity=0.6, strokeWidth=1.5).encode(
                    x=alt.X('Time', title='Années'),
                    y=alt.Y('Prix', title='Prix du Sous-Jacent', scale=alt.Scale(zero=False)),
                    color=alt.Color('Path:N', legend=None),
                    tooltip=['Time', 'Prix', 'Path']
                ).properties(height=380, title="Paths (LSM)").interactive()
                rule = alt.Chart(pd.DataFrame({'y': [float(option_strike)]})).mark_rule(color='black', strokeWidth=2, strokeDash=[5,5]).encode(y='y')
                st.altair_chart(chart_lsm + rule, use_container_width=True)

            if exercise_style == "European" and ("Heston" in pricing_method) and h_show_paths:
                st.subheader("Visualisation des Trajectoires (Heston)")
                st.caption("Trajectoires du spot et de la variance simulées sous Heston.")

                heston_model = HestonMonteCarloModel(chunk_size=20000)
                df_S, df_v = heston_model.simulate_paths_for_visualization(
                    S0=float(spot_price),
                    T=float(option_maturity),
                    r=float(option_risk_free_rate),
                    q=float(dividend_yield),
                    v0=float(heston_params["v0"]),
                    kappa=float(heston_params["kappa"]),
                    theta=float(heston_params["theta"]),
                    xi=float(heston_params["xi"]),
                    rho=float(heston_params["rho"]),
                    n_paths_visu=30,
                    steps=int(h_steps),
                    seed=int(h_seed),
                    full_truncation=True,
                )

                tabS, tabV = st.tabs(["Trajectoires S", "Trajectoires v"])
                with tabS:
                    dfS_long = df_S.reset_index(names="Time").melt("Time", var_name="Path", value_name="S")
                    chartS = alt.Chart(dfS_long).mark_line(opacity=0.6).encode(
                        x="Time",
                        y=alt.Y("S", scale=alt.Scale(zero=False)),
                        color=alt.Color("Path:N", legend=None),
                        tooltip=["Time", "S", "Path"]
                    ).properties(height=350).interactive()
                    st.altair_chart(chartS, use_container_width=True)

                with tabV:
                    dfV_long = df_v.reset_index(names="Time").melt("Time", var_name="Path", value_name="v")
                    chartV = alt.Chart(dfV_long).mark_line(opacity=0.6).encode(
                        x="Time",
                        y=alt.Y("v", scale=alt.Scale(zero=False)),
                        color=alt.Color("Path:N", legend=None),
                        tooltip=["Time", "v", "Path"]
                    ).properties(height=350).interactive()
                    st.altair_chart(chartV, use_container_width=True)

            # =========================================================
            # 6) Grecques (BS exactes, MC approx / American FD approx)
            # =========================================================
            st.subheader("Grecques de l'Option")

            if exercise_style == "European" and ("Black-Scholes" in pricing_method):
                bs_model = BlackScholesModel()
                greeks_val = {
                    "Delta": bs_model.delta(option),
                    "Gamma": bs_model.gamma(option),
                    "Vega": bs_model.vega(option),
                    "Theta": bs_model.theta(option),
                    "Rho": bs_model.rho(option)
                }
                cols = st.columns(5)
                for i, (k, v) in enumerate(greeks_val.items()):
                    cols[i].metric(k, f"{v:.4f}")

                # --- ÉVOLUTION DES GRECQUES (comme dans CodeAnto1) ---
                with st.expander("📈 Évolution des grecques (Black-Scholes)", expanded=True):
                    st.caption("Courbes des grecques en fonction du spot S (autour du strike K). Trait rouge = K, trait gris = spot actuel.")

                    range_pct = st.slider(
                        "Amplitude autour du strike (± en proportion)",
                        min_value=0.05, max_value=1.00, value=0.50, step=0.05,
                        key="ov_greeks_range_pct"
                    )
                    steps_g = st.slider(
                        "Nombre de points pour les courbes",
                        min_value=20, max_value=200, value=80, step=10,
                        key="ov_greeks_range_steps"
                    )

                    greeks_data = bs_model.compute_greeks_range(option, range_pct=float(range_pct), steps=int(steps_g))
                    df_greeks = pd.DataFrame(greeks_data)

                    greeks_config = [
                        {"label": "Delta (Δ)", "value": bs_model.delta(option), "key": "Delta"},
                        {"label": "Gamma (Γ)", "value": bs_model.gamma(option), "key": "Gamma"},
                        {"label": "Vega (ν)",  "value": bs_model.vega(option),  "key": "Vega"},
                        {"label": "Theta (Θ)", "value": bs_model.theta(option), "key": "Theta"},
                        {"label": "Rho (ρ)",   "value": bs_model.rho(option),   "key": "Rho"},
                    ]

                    strike_line = alt.Chart(pd.DataFrame({"x": [float(option_strike)]})).mark_rule(
                        color="red", strokeDash=[5, 5]
                    ).encode(x="x:Q")

                    spot_line = alt.Chart(pd.DataFrame({"x": [float(spot_price)]})).mark_rule(
                        color="gray", strokeDash=[2, 2]
                    ).encode(x="x:Q")

                    for item in greeks_config:
                        col_metric, col_chart = st.columns([1, 2])

                        with col_metric:
                            st.metric(item["label"], f"{item['value']:.4f}")

                        with col_chart:
                            greek_key = item["key"]
                            chart_line = alt.Chart(df_greeks).mark_line().encode(
                                x=alt.X("S:Q", title="Spot (S)"),
                                y=alt.Y(f"{greek_key}:Q", title=None),
                                tooltip=[
                                    alt.Tooltip("S:Q", format=",.2f"),
                                    alt.Tooltip(f"{greek_key}:Q", format=",.4f")
                                ]
                            )
                            final_chart = (chart_line + strike_line + spot_line).properties(
                                height=200,
                                title=f"{greek_key}"
                            ).interactive()
                            st.altair_chart(final_chart, use_container_width=True)

                        st.markdown("<hr style='margin: 0px; opacity: 0.2;'>", unsafe_allow_html=True)

            elif exercise_style == "European" and ("Monte Carlo" in pricing_method and "GBM" in pricing_method):
                with st.spinner("Calcul des Grecques par chocs (GBM MC)..."):
                    delta_mc, gamma_mc, vega_mc, theta_mc, rho_mc = mc_model.Greeks_diff_MC(
                        option,
                        h=spot_price * 0.01,
                        h_vol=0.001,        # 10 bps vol
                        h_r=1e-4,           # 1 bp rate
                        h_T=1/365.25        # 1 day
                    )

                g1, g2, g3, g4, g5 = st.columns(5)
                g1.metric("Delta (MC)", f"{delta_mc:.4f}")
                g2.metric("Gamma (MC)", f"{gamma_mc:.4f}")
                g3.metric("Vega (MC)", f"{vega_mc:.4f}")
                g4.metric("Theta (MC)", f"{theta_mc:.4f}")
                g5.metric("Rho (MC)", f"{rho_mc:.4f}")

            elif exercise_style == "European" and ("Heston" in pricing_method):
                st.info("Greeks Heston : non implémentées (bumping Heston possible, mais coûteux et bruité).")

            else:
                # American: finite-difference greeks (tree fast, LSM slower)
                bumpS = float(spot_price) * 0.01
                bumpV = 0.01
                bumpR = 1e-4

                def _price_american_given(S0_, r_, sigma_):
                    if "Binomial Tree" in pricing_method:
                        return price_american_crr(
                            S0=float(S0_), K=float(option_strike), T=float(option_maturity),
                            r=float(r_), q=float(dividend_yield), sigma=float(sigma_),
                            steps=int(steps_tree if steps_tree is not None else 800),
                            opt_type=str(option_type).lower()
                        )
                    # LSM
                    p, _ = price_american_lsm(
                        S0=float(S0_), K=float(option_strike), T=float(option_maturity),
                        r=float(r_), q=float(dividend_yield), sigma=float(sigma_),
                        n_paths=int(n_paths_lsm if n_paths_lsm is not None else 40000),
                        steps=int(steps_lsm if steps_lsm is not None else 100),
                        opt_type=str(option_type).lower(),
                        degree=int(degree_lsm if degree_lsm is not None else 2),
                        seed=123,
                        antithetic=bool(antithetic_lsm if antithetic_lsm is not None else True),
                    )
                    return p

                with st.spinner("Calcul approx des grecques (différences finies)..."):
                    p0 = float(price_result_option)
                    p_up = _price_american_given(spot_price + bumpS, option_risk_free_rate, option_volatility)
                    p_dn = _price_american_given(spot_price - bumpS, option_risk_free_rate, option_volatility)
                    delta_fd = (p_up - p_dn) / (2 * bumpS)
                    gamma_fd = (p_up - 2*p0 + p_dn) / (bumpS**2)

                    p_v = _price_american_given(spot_price, option_risk_free_rate, option_volatility + bumpV)
                    vega_fd = (p_v - p0) / bumpV

                    p_r = _price_american_given(spot_price, option_risk_free_rate + bumpR, option_volatility)
                    rho_fd = (p_r - p0) / bumpR

                g1, g2, g3, g4 = st.columns(4)
                g1.metric("Delta (FD)", f"{delta_fd:.4f}")
                g2.metric("Gamma (FD)", f"{gamma_fd:.4f}")
                g3.metric("Vega (FD)", f"{vega_fd:.4f}")
                g4.metric("Rho (FD)", f"{rho_fd:.4f}")
                st.caption("FD = différences finies. Sur LSM, ces grecques peuvent être bruitées (augmenter paths si besoin).")

            # =========================================================
            # 7) Payoff graph
            # =========================================================
            st.subheader("Profil de Payoff à Maturité")
            payoff_data = option.payoff_graph()
            st.line_chart(payoff_data.set_index('spots'), height=200)

        except Exception as e:
            st.error(f"Erreur lors du calcul : {e}")

    
    # 5. Option Asiatique

    elif instrument_type == "Option Asiatique":
        st.header(f"Paramètres & Théorie : {instrument_type}")
        
        # --- PARTIE PÉDAGOGIQUE (LE COURS) ---
        with st.expander("📘 Cours : Comprendre l'Option Asiatique & Monte Carlo", expanded=True):
            st.markdown("""
            ### 1. Qu'est-ce qu'une Option Asiatique ?
            Une option asiatique est une option dont le payoff dépend de la **moyenne** des prix du sous-jacent sur la période, et non du prix final.
            Elle est moins chère qu'une option vanille car la moyenne est moins volatile que le prix instantané.
            
            **Le Payoff (Call Arithmétique) :**
            """)
            st.latex(r"Payoff = \max\left( \frac{1}{N} \sum_{i=1}^{N} S_{t_i} - K, 0 \right)")
            
            st.markdown("""
            ### 2. Pourquoi Monte Carlo ?
            La somme de variables Log-Normales (les prix $S_t$) n'est pas Log-Normale. 
            Il n'existe donc **pas de formule fermée de Black-Scholes** pour la moyenne arithmétique.
            Nous devons simuler des milliers de trajectoires possibles du prix de l'action.
            
            **La Dynamique du Sous-jacent (GBM) :**
            """)
            st.latex(r"S_{t+\Delta t} = S_t \exp\left( (r - \frac{\sigma^2}{2})\Delta t + \sigma \sqrt{\Delta t} Z \right)")
            st.caption("Où $Z$ est une variable aléatoire Normale Standard $\mathcal{N}(0,1)$.")
            
            st.markdown("""
            ### 3. L'Astuce de "Quant" : La Variable de Contrôle
            Monte Carlo est lent et bruité. Pour accélérer la précision, nous utilisons une **Variable de Contrôle**.
            Nous connaissons le prix exact de l'Option Asiatique à moyenne *Géométrique* (formule analytique).
            
            **L'Estimateur corrigé :**
            """)
            st.latex(r"Prix_{Final} = Prix_{MC}^{Arith} - \beta \left( Prix_{MC}^{Geo} - Prix_{Exact}^{Geo} \right)")
            st.caption("Cette technique permet de réduire la variance (l'erreur) d'un facteur 100 à 1000 !")

        st.divider()
        # --- EXEMPLE CONCRET : ASIATIQUE ---
        with st.expander("Un Exemple concret (Option Asiatique)", expanded=False):
            st.markdown("""
            ### Exemple
            Suppose que l’option regarde 5 observations de prix :
            [100, 102, 98, 101, 99]
            Moyenne = 100.0

            - Si $K=100$ et que c’est un Call asiatique (moyenne arithmétique) :
            $payoff = max(100 - 100, 0) = 0$

            Comparaison :
            - Une option vanille regarderait seulement $$S_T$$ (le dernier prix).
            - L’asiatique “lisse” les pics → souvent **moins chère**.
            """)

        # --- INPUTS ---
        col_asia1, col_asia2, col_asia3 = st.columns(3)
        
        with col_asia1:
            selected_ticker_label = st.selectbox("Sous-jacent", EQUITY_TICKERS)
            selected_ticker = selected_ticker_label.split(" ")[0]
            try:
                spot_price = DATA_LOADER.get_equity_spot(selected_ticker)
                st.success(f"Spot ($S_0$): {spot_price:.2f} €")
            except:
                spot_price = 100.0
                st.warning("Prix par défaut utilisé.")

            maturity = st.number_input("Maturité (Années)", 0.1, 5.0, 1.0, 0.1)

        with col_asia2:
            strike = st.number_input(
                "Strike (K)",
                min_value=0.0,
                max_value=None,
                value=float(spot_price),   # <-- default = spot
                step=5.0
            )
            volatility = st.number_input("Volatilité ($\sigma$ %)", 1.0, 200.0, 20.0, 1.0) / 100
            rate_source = st.radio("Taux sans risque", ["Manuel", "Marché (FRED)"])
            
            if rate_source == "Manuel":
                r = st.number_input("Taux sans risque ($r$ %)", 0.0, 20.0, 3.0) / 100
            else:
                r = rf_from_curve(maturity)
                st.info(f"Taux marché : {r*100:.2f} %")

        with col_asia3:
            option_type = st.selectbox("Type", ["Call", "Put"])
            n_simulations = st.slider("Nombre de Simulations (MC)", 1000, 100000, 10000, 1000)
            obs_steps = st.number_input("Nb d'observations (N)", 10, 252*2, 252, help="252 = Moyenne quotidienne sur 1 an")

        # --- CALCUL ---
        if st.button("Lancer le Pricing Asiatique"):
            try:
                # 1. Objets
                params = Parametres(maturity=maturity, risk_free_rate=r, strike=strike, volatility=volatility)
                asian_opt = OptionAsiatique(params, spot=spot_price, option_type=option_type, observation_steps=obs_steps)
                model = AsianPricingModel(num_simulations=n_simulations)
                
                # 2. Pricing
                with st.spinner("Simulation de Monte Carlo & Optimisation Control Variate..."):
                    price, var_reduction = model.price_option_mc_control_variate(asian_opt)
                
                # 3. Affichage Résultats
                st.markdown("### 🎯 Résultats du Pricing")
                c1, c2, c3 = st.columns(3)
                c1.metric("Prix Option Asiatique", f"{price:.4f} €")
                c2.metric("Réduction de Variance", f"{var_reduction*100:.2f} %", delta="Précision Extreme")
                c3.metric("Type de Moyenne", "Arithmétique Discrète")
                
            
                # --- Greeks (Asian, MC bumping) ---
                with st.spinner("Calcul des Grecques (MC bumping)..."):
                    greeks = model.greeks_mc_control_variate(
                        asian_opt,
                        hS=spot_price * 0.01,   # 1% spot
                        hV=0.01,                # +1 vol point
                        hR=1e-4,                # +1 bp
                        hT=1/365.25,            # +1 day
                        seed=123
                    )

                st.subheader("Grecques (Option Asiatique)")
                g1, g2, g3, g4, g5 = st.columns(5)
                g1.metric("Delta", f"{greeks['Delta']:.4f}")
                g2.metric("Gamma", f"{greeks['Gamma']:.6f}")
                g3.metric("Vega (1% vol)", f"{greeks['Vega_1pp']:.4f}")
                g4.metric("Theta (1 day)", f"{greeks['Theta_1day']:.4f}")
                g5.metric("Rho (1 bp)", f"{greeks['Rho_1bp']:.4f}")



                
                # 4. Visualisation Pédagogique (Trajectoires)
                st.markdown("### 📉 Visualisation :")
                st.caption(f"Voici 50 trajectoires simulées du sous-jacent {selected_ticker} sur {maturity} an(s).")
                st.caption("L'option asiatique 'regarde' le prix à chaque pas de temps pour calculer sa moyenne.")
                
                paths_df = model.get_paths_for_visualization(asian_opt, num_paths_to_plot=50)
                
                # Graphique avec Altair pour un rendu fin
                paths_melted = paths_df.reset_index().melt('index', var_name='Chemin', value_name='Prix')

                chart = alt.Chart(paths_melted).mark_line(opacity=0.6, strokeWidth=3).encode(
                    x=alt.X('index:Q', title='Temps (Années)'),
                    y=alt.Y('Prix:Q', title='Prix du Sous-Jacent (€)', scale=alt.Scale(zero=False)),
                    detail='Chemin:N',                 # IMPORTANT : une ligne par trajectoire
                    color=alt.value('#00E5FF'),        # couleur très visible sur fond sombre
                    tooltip=[
                        alt.Tooltip('index:Q', title='Temps', format='.3f'),
                        alt.Tooltip('Prix:Q', title='Prix', format=',.2f'),
                        alt.Tooltip('Chemin:N', title='Chemin')
                    ]
                ).properties(height=400)

                # Ajout de la ligne du Strike
                strike_line = alt.Chart(pd.DataFrame({'y': [strike]})).mark_rule(
                    color='red', strokeDash=[5,5]
                ).encode(y='y')

                st.altair_chart(chart + strike_line, use_container_width=True)

                
                st.info("La ligne rouge pointillée représente le Strike (K). Pour un Call, on gagne si la **moyenne** des courbes bleues est au-dessus de la ligne rouge.")

            except Exception as e:
                st.error(f"Erreur de calcul : {e}")

    # 6. Option Barrière
    elif instrument_type == "Option Barrière":
        st.header(f"Paramètres & Théorie : {instrument_type}")

        # Choix de la méthode de pricing
        pricing_method = st.radio(
            "Méthode de Valorisation", 
            ["Monte Carlo Avancé (Pont Brownien)", "Formule Fermée (Merton / Reiner-Rubinstein)"],
            help="Monte Carlo est plus lent mais plus flexible. La formule fermée est instantanée mais rigide."
        )

        # --- PARTIE PÉDAGOGIQUE (LE COURS) ---
        if pricing_method == "Monte Carlo Avancé (Pont Brownien)":
            with st.expander("📘 Cours : Le Challenge des Barrières & Le Pont Brownien", expanded=True):
                st.markdown("""
                ### 1. Le Problème de la Discrétisation
                Une option barrière s'active (In) ou se désactive (Out) si le sous-jacent touche un niveau $B$.
                Dans une simulation Monte Carlo classique, on ne regarde le prix qu'à des instants précis $t_1, t_2, ...$.
                
                **Le Danger :** Le prix peut très bien franchir la barrière *entre* deux pas de temps ($t_i$ et $t_{i+1}$) et revenir en arrière.
                Si on ne vérifie que les points discrets, on rate cet événement. On sous-estime le risque.
                """)
                
                st.markdown("""
                ### 2. La Solution : Le Pont Brownien (Brownian Bridge)
                Pour corriger ce biais, nous utilisons une formule probabiliste issue du **Principe de Réflexion** du mouvement brownien.
                Sachant les positions $S_t$ et $S_{t+1}$, la probabilité d'avoir touché la barrière $B$ dans l'intervalle est :
                """)
                
                st.latex(r"P(\text{Hit}) = \exp\left( -\frac{2 \ln(S_t/B) \cdot \ln(S_{t+1}/B)}{\sigma^2 \Delta t} \right)")
                
                st.markdown(r"""
                ### 3. Algorithme de Simulation Avancé
                Pour chaque pas de temps de chaque trajectoire :
                1.  **Vérification Discrète :** Est-ce que $S_{t+1}$ a franchi la barrière ?
                2.  **Vérification Continue :** Si non, on tire un nombre aléatoire $U \sim \mathcal{U}[0,1]$.
                    * Si $U < P(\text{Hit})$, alors on considère que la barrière a été touchée "entre les lignes".
                """)

        elif pricing_method == "Formule Fermée (Merton / Reiner-Rubinstein)":
            with st.expander("📘 Cours : Les Formules Analytiques (Reiner-Rubinstein)", expanded=True):
                st.markdown(r"""
                ### 1. Le Principe de Réflexion (Mirror Principle)
                Le modèle analytique repose sur l'idée géométrique que pour toute trajectoire touchant une barrière $H$, il existe une trajectoire "miroir" symétrique par rapport à $H$.
                Le prix d'une option barrière s'écrit donc comme une **combinaison de quatre briques** (notations de Haug), avec $\phi = +1$ pour un Call, $-1$ pour un Put, et $\eta = +1$ pour une barrière Down, $-1$ pour une barrière Up :
                * **$A$** : l'option vanille (Black-Scholes classique) ;
                * **$B$** : la vanille "coupée" au niveau de la barrière ;
                * **$C$** : l'option **reflet** (trajectoires miroir, terme en $\ln(H^2/SK)$) ;
                * **$D$** : le reflet "coupé" au niveau de la barrière.
                """)

                st.latex(r"C = \phi S e^{-qT} \left(\frac{H}{S}\right)^{2(\mu+1)} \mathcal{N}(\eta y_1) - \phi K e^{-rT} \left(\frac{H}{S}\right)^{2\mu} \mathcal{N}(\eta y_1 - \eta\sigma\sqrt{T})")

                st.markdown(r"""
                Où $\mu = (r - q - \sigma^2/2)/\sigma^2$ et $y_1 = \dfrac{\ln(H^2/SK)}{\sigma\sqrt{T}} + (1+\mu)\sigma\sqrt{T}$.

                ### 2. Quelle combinaison ? Cela dépend de la position du strike
                Prix des options **Knock-In** (activantes) :
                """)

                st.markdown(r"""
                | Option | $K > H$ | $K \le H$ |
                |---|---|---|
                | Down-and-In Call | $C$ | $A - B + D$ |
                | Up-and-In Call | $A$ | $B - C + D$ |
                | Down-and-In Put | $B - C + D$ | $A$ |
                | Up-and-In Put | $A - B + D$ | $C$ |
                """)

                st.markdown(r"""
                Le cas "simple" $V_{In} = C$ (le reflet) n'est vrai que pour le **Down-and-In Call avec $K > H$** et l'**Up-and-In Put avec $K \le H$**.
                Dès que le strike est "du mauvais côté" de la barrière, il faut corriger avec $B$ et $D$.

                ### 3. La parité In / Out
                Une option Out se déduit de l'option In : à chaque trajectoire, exactement une des deux est vivante à maturité.
                """)
                st.latex(r"V_{Out} = V_{Vanille} - V_{In}")
                st.markdown(r"*Exemple : un Up-and-Out Call avec $K < H$ vaut beaucoup moins que la vanille, car les trajectoires qui finissent très dans la monnaie ont presque toutes touché la barrière.*")

        st.divider()
        # --- EXEMPLES SIMPLES : BARRIÈRES ---
        with st.expander("🧠 Exemples simples (Barrières)", expanded=False):
            st.markdown("""
            ### 1) Up-and-Out Call (barrière désactivante)
            - Tu as un Call, MAIS si le prix touche la barrière B à n’importe quel moment → l’option devient 0.
            - Exemple : K=100, B=140  
            Même si à la fin $S_T=160$, si le chemin a touché 140 avant → payoff = 0.

            ### 2) Up-and-In Call (barrière activante)
            - L’option n’existe “vraiment” que si le prix touche B.
            - Exemple : K=100, B=140  
            Si le prix ne touche jamais 140 → payoff = 0 même si $S_T=160$.

            ### 3) Pourquoi le Brownian Bridge ?
            Même si tu simules à dates discrètes, le prix peut franchir la barrière entre deux pas :
            le pont brownien corrige ce biais.
            """)


        # --- INPUTS ---
        st.subheader("Configuration du Produit")
        
        col_b1, col_b2, col_b3 = st.columns(3)

        with col_b1:
            # Actif
            selected_ticker_label = st.selectbox("Sous-jacent", EQUITY_TICKERS)
            selected_ticker = selected_ticker_label.split(" ")[0]
            try:
                spot_price = DATA_LOADER.get_equity_spot(selected_ticker)
                st.success(f"Spot ($S_0$): {spot_price:.2f} €")
            except:
                spot_price = 100.0
                st.warning("Prix par défaut utilisé.")
            
            maturity = st.number_input("Maturité (Années)", 0.1, 5.0, 1.0, 0.1)

        with col_b2:
            # Paramètres de marché
            volatility = st.number_input("Volatilité ($\sigma$ %)", 1.0, 200.0, 20.0, 1.0) / 100
            rate_source = st.radio("Taux sans risque", ["Manuel", "Marché (FRED)"])
            if rate_source == "Manuel":
                r = st.number_input("Taux sans risque ($r$ %)", 0.0, 20.0, 3.0) / 100
            else:
                r = rf_from_curve(maturity)
                st.info(f"Taux marché : {r*100:.2f} %")

        with col_b3:
            # Option de base
            option_type = st.selectbox("Type Option (Payoff)", ["Call", "Put"])
            # Correction Strike Max Value
            strike = st.number_input("Strike ($K$)", min_value=0.0, max_value=None, value=spot_price, step=5.0)

        st.divider()
        st.subheader("Paramètres de la Barrière")

        col_bar1, col_bar2, col_bar3 = st.columns(3)
        
        with col_bar1:
            barrier_level = st.number_input("Niveau Barrière ($B$)", min_value=0.0, max_value=None, value=spot_price * 1.2, step=5.0)
        
        with col_bar2:
            direction = st.selectbox("Direction de la Barrière", ["Up (Haut)", "Down (Bas)"])
            style = st.selectbox("Mécanisme", ["Out (Désactivante)", "In (Activante)"])
            
            # Traduction pour le code (ex: "Up (Haut)" -> "up")
            dir_code = "up" if "Up" in direction else "down"
            style_code = "out" if "Out" in style else "in"
            barrier_type_str = f"{dir_code}-{style_code}" # ex: "up-out"

        with col_bar3:
            if pricing_method == "Monte Carlo Avancé (Pont Brownien)":
                n_simulations = st.slider("Simulations (MC)", 1000, 100000, 10000, 1000)
                n_steps = st.number_input("Pas de temps (Steps)", 10, 500, 252, help="252 = 1 jour ouvré")
            else:
                st.info("Pas de simulation requise pour la formule fermée.")

        # --- VÉRIFICATION DE COHÉRENCE ---
        is_consistent = True
        if dir_code == "up" and barrier_level < spot_price:
            st.error(f"⛔ Incohérence : Une barrière 'Up' ({barrier_level}) doit être supérieure au Spot ({spot_price}).")
            is_consistent = False
        if dir_code == "down" and barrier_level > spot_price:
            st.error(f"⛔ Incohérence : Une barrière 'Down' ({barrier_level}) doit être inférieure au Spot ({spot_price}).")
            is_consistent = False

        # --- BOUTON DE CALCUL ---
        if st.button("Lancer le Pricing Barrière", disabled=not is_consistent):
            
            try:
                # 1. Instanciation des objets
                params = Parametres(maturity=maturity, risk_free_rate=r, strike=strike, volatility=volatility)
                
                barrier_opt = BarrierOption(
                    parameters=params,
                    spot=spot_price,
                    option_type=option_type,
                    barrier=barrier_level,
                    barrier_type=barrier_type_str
                )
                
                model = BarrierPricingModel(num_simulations=10000) # Simulations par défaut si non utilisé
                
                price = 0.0
                
                # 2. Pricing selon méthode
                if pricing_method == "Formule Fermée (Merton / Reiner-Rubinstein)":
                    with st.spinner("Calcul Analytique (Reiner-Rubinstein)..."):
                        price = model.price_barrier_analytic(barrier_opt)
                        st.success("Calcul effectué par formule exacte.")
                        
                else:
                    # Monte Carlo
                    model.num_simulations = n_simulations # Mise à jour avec slider
                    with st.spinner("Simulation Monte Carlo avec Correction Pont Brownien..."):
                        price = model.price_barrier_option_mc(barrier_opt, steps=n_steps)
                
                # 3. Affichage Résultat
                st.markdown("### 🎯 Résultat")
                col_res1, col_res2 = st.columns(2)

                with col_res1:
                    st.metric(f"Prix Option {barrier_type_str}", f"{price:.4f} €")

                with col_res2:
                    # Petit message explicatif dynamique
                    if style_code == "out":
                        st.caption(f"ℹ️ L'option vaut 0 € si le sous-jacent touche **{barrier_level} €** avant {maturity} ans.")
                    else:
                        st.caption(f"ℹ️ L'option ne s'active (et ne vaut quelque chose) QUE si le sous-jacent touche **{barrier_level} €**.")

                # --- GRECQUES (EN DEHORS DES COLONNES) ---
                st.divider()
                st.subheader("Grecques (Option Barrière)")

                # Choix cohérent de la méthode
                greek_method = "analytic" if "Formule Fermée" in pricing_method else "mc"
                greek_steps = n_steps if greek_method == "mc" else 252

                with st.spinner("Calcul des grecques (différences finies)..."):
                    greeks_barrier = model.compute_greeks_fd(
                        barrier_opt,
                        method=greek_method,
                        steps=greek_steps,
                        spot_bump_pct=0.01,
                        vol_bump=0.01,
                        rate_bump=1e-4,
                        time_bump=1/365.25,
                        seed=123
                    )

                cols_g = st.columns(5)
                cols_g[0].metric("Delta", f"{greeks_barrier['Delta']:.4f}")
                cols_g[1].metric("Gamma", f"{greeks_barrier['Gamma']:.4f}")
                cols_g[2].metric("Vega (€/1% vol)",  f"{greeks_barrier['Vega']:.4f}")
                cols_g[3].metric("Theta (€/jour)", "N/A" if greeks_barrier["Theta"] is None else f"{greeks_barrier['Theta']:.4f}")
                cols_g[4].metric("Rho (€/1bp)",   f"{greeks_barrier['Rho']:.4f}")

                st.caption(
                    "Unités : Vega est affiché par +1% de volatilité, Rho par +1 bp de taux, Theta par jour. "
                    "Note : pour les barrières, les grecques (surtout Gamma/Theta) peuvent être bruitées en Monte Carlo — augmentez les simulations si besoin."
                )

            except Exception as e:
                st.error(f"Erreur de calcul : {e}")

def page_calibration():
    st.header("Calibration de la Volatilité Implicite")
    
    # --- PARTIE PÉDAGOGIQUE ---
    with st.expander("📘 Cours : Comprendre la Volatilité Implicite & les Algorithmes", expanded=True):
        st.markdown("""
        ### 1. Qu'est-ce que la Volatilité Implicite ?
        Contrairement à la volatilité historique (basée sur le passé), la volatilité implicite est une **anticipation du marché**.
        C'est la valeur de $\sigma$ qu'il faut injecter dans la formule de Black-Scholes pour retrouver exactement le prix de marché de l'option.
        
        * Si le prix de marché monte (demande forte), la volatilité implicite monte.
        * C'est souvent considéré comme le "thermomètre de la peur" ou de l'incertitude des investisseurs.

        ### 2. Le Problème Mathématique
        La formule de Black-Scholes $C(S, K, r, T, \sigma)$ n'est pas inversible analytiquement. On ne peut pas isoler $\sigma$.
        Nous cherchons la racine de la fonction :
        """)
        st.latex(r"f(\sigma) = C_{Black-Scholes}(\sigma) - Prix_{Marché} = 0")
        
        st.markdown("""
        ### 3. Les Algorithmes de Résolution Numérique
        Nous utilisons deux méthodes classiques pour trouver cette racine :
        
        **A. Méthode de Newton-Raphson (La Rapide)** \n
        Elle utilise la dérivée de la fonction (le Vega) pour converger très vite.
        """)
        st.latex(r"\sigma_{n+1} = \sigma_n - \frac{f(\sigma_n)}{f'(\sigma_n)} = \sigma_n - \frac{C(\sigma_n) - P_{Marché}}{\nu(\sigma_n)}")
        st.caption("Avantage : Très rapide. Inconvénient : Peut diverger si le Vega est proche de 0 (option très en dehors de la monnaie).")
        
        st.markdown("""
        **B. Méthode de Dichotomie (La Robuste)**
        Elle coupe l'intervalle de recherche en deux à chaque étape.
        """)
        st.markdown("* Si $f(milieu) > 0$, on cherche dans la moitié gauche.")
        st.markdown("* Si $f(milieu) < 0$, on cherche dans la moitié droite.")
        st.caption("Avantage : Converge toujours si la solution existe. Inconvénient : Plus lent.")

    st.divider()

    # --- SÉLECTION DES PARAMÈTRES ---
    st.subheader("Configuration du Market Data")
    
    # Ligne 1 : Actif et Maturité
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_ticker_label = st.selectbox("Actif Sous-jacent", EQUITY_TICKERS[9:]) # Actions uniquement
        selected_ticker = selected_ticker_label.split(" ")[0]
        try:
            S0 = DATA_LOADER.get_equity_spot(selected_ticker)
            st.success(f"Spot ($S_0$) : **{S0:,.2f} €**")
        except:
            S0 = 100.0
            st.warning("Spot par défaut utilisé")

    with col2:
        try:
            expiration_choice = DATA_LOADER.get_option_expirations(selected_ticker)
            selected_expiration = st.selectbox("Date d'Échéance", expiration_choice[1:], index=0)
            
            # Calcul Maturité
            delta = datetime.strptime(selected_expiration, "%Y-%m-%d") - datetime.now()
            maturity = delta.days / 365.25
            st.info(f"Maturité ($T$) : **{maturity:.4f} ans**")
        except:
            st.error("Pas de données d'options disponibles.")
            return

    with col3:
        rate_source = st.radio("Source Taux sans risque", ["Manuel", "Automatique (Courbe)"], horizontal=True)
        if rate_source == "Manuel":
            risk_free_rate = st.number_input("Taux ($r$ %)", 0.0, 20.0, 4.0) / 100
        else:
            risk_free_rate = rf_from_curve(maturity)
            st.info(f"Taux interpolé : **{risk_free_rate*100:.3f} %**")

    # Ligne 2 : Option Spécifique
    col4, col5, col6 = st.columns(3)
    
    with col4:
        option_type = st.selectbox("Type d'Option", ["call", "put"])
    
    with col5:
        # Gestion du Strike
        # On essaie de récupérer la chaîne pour proposer des strikes valides si possible, sinon libre
        try:
            chain = DATA_LOADER.get_option_chain_data(selected_ticker, selected_expiration)
            df_opt = chain['calls'] if option_type == 'call' else chain['puts']
            valid_strikes = df_opt['strike'].tolist()
            # Strike le plus proche du spot par défaut
            default_idx = min(range(len(valid_strikes)), key=lambda i: abs(valid_strikes[i]-S0))
            option_strike = st.selectbox("Strike ($K$)", valid_strikes, index=default_idx)
        except:
            option_strike = st.number_input("Strike ($K$)", value=S0)

    with col6:
        # Récupération automatique du prix de marché
        try:
            market_price_auto = DATA_LOADER.get_market_price_interpolated(
                selected_ticker, 
                expiration_date=selected_expiration, 
                strike=option_strike, 
                option_type=option_type
            )
            # Utilisation de st.metric pour un rendu natif et lisible
            st.metric(
                label="Prix de Marché (Cible)",
                value=f"{market_price_auto:.2f} €",
                help="Prix récupéré via Yahoo Finance (interpolé)"
            )
            market_price = market_price_auto
        except:
            # Si échec, on propose la saisie manuelle
            market_price = st.number_input("Prix de Marché (Manuel)", value=10.0)

    st.divider()

    # --- CALIBRATION ---
    st.subheader("Algorithme de Calibration")

    calibration_method = st.radio(
        "Méthode Numérique",
        ["Newton-Raphson (Gradient)", "Bisection (Dichotomie)"]
    )

    if st.button("Lancer la Calibration", type="primary"):

        params = Parametres(
            maturity=maturity,
            risk_free_rate=risk_free_rate,
            strike=option_strike,
            volatility=0.2
        )
        option = OptionVanille(parameters=params, spot=S0, option=option_type)
        bs_model = BlackScholesModel()
        solver = ImpliedVolatility(market_price, option, bs_model)

        start_time = datetime.now()
        implied_vol = solver.newton_raphson() if "Newton" in calibration_method else solver.dichotomie()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() * 1000  # ms

        if implied_vol is not None and 0 < implied_vol < 5:
            st.success(f"Calibration réussie en {duration:.1f} ms")

            solved_vol = float(implied_vol)
            option.parameters.volatility = solved_vol

            price_recalc = bs_model.price_option(option)
            vega_recalc  = bs_model.vega(option)

            c_res1, c_res2, c_res3 = st.columns(3)
            c_res1.metric("Volatilité Implicite", f"{solved_vol*100:.2f} %")
            c_res2.metric("Prix Re-calculé", f"{price_recalc:.2f} €", delta=f"{price_recalc - market_price:+.2f} €")
            c_res3.metric("Vega (Sensibilité)", f"{vega_recalc:.4f}")

            vol_range = np.linspace(max(0.01, solved_vol - 0.3), solved_vol + 0.3, 50)
            prices = []
            for v in vol_range:
                option.parameters.volatility = float(v)
                prices.append(bs_model.price_option(option))

            option.parameters.volatility = solved_vol  # restore

            df_chart = pd.DataFrame({"Volatilité": vol_range, "Prix Théorique": prices})

            base_chart = alt.Chart(df_chart).mark_line().encode(
                x=alt.X("Volatilité", axis=alt.Axis(format="%")),
                y="Prix Théorique"
            )

            market_line = alt.Chart(pd.DataFrame({"y": [market_price]})).mark_rule(
                color="red", strokeDash=[5, 5]
            ).encode(y="y")

            point = alt.Chart(pd.DataFrame({"x": [solved_vol], "y": [market_price]})).mark_point(
                color="green", size=100, filled=True
            ).encode(x="x", y="y")

            st.altair_chart((base_chart + market_line + point).properties(
                title="Visualisation de la Solution (Volatilité vs Prix)"
            ), use_container_width=True)

        else:
            st.error("Échec de la calibration. Vérifiez si le prix de marché est cohérent (pas d'arbitrage).")


def page_surface():
    st.header("Analyse de Surface de Volatilité")

    # =========================
    # PARTIE COURS (pédagogie)
    # =========================
    st.markdown(r"""
### C’est quoi une *surface de volatilité implicite* ?

La **volatilité implicite** (IV) est la volatilité $$\sigma$$ qui rend le prix théorique (souvent via Black–Scholes)
**égal au prix observé sur le marché** :

$$
C_{\text{BS}}(S_0,K,T,r,q,\sigma)=C_{\text{mkt}}
\qquad\text{(pour un call)}
$$

Autrement dit, on **inverse** le modèle : on cherche $$\sigma$$ telle que la différence
$$
f(\sigma)=C_{\text{BS}}(\sigma)-C_{\text{mkt}}=0
$$
soit nulle (ou très proche de 0).

---

### Pourquoi une “surface” ?
Parce que, dans la pratique, l’IV dépend à la fois :
- de la **maturité** \(T\),
- du **strike** \(K\),

donc on obtient une fonction à deux variables :

$$
\sigma_{\text{impl}} = \sigma_{\text{impl}}(K,T)
$$

et on représente cette fonction en 3D (ou en heatmap).

---

### Comment lire le graphe ?
- Axe **X** : le **strike** \(K\)
- Axe **Y** : la **maturité** \(T\)
- Axe **Z** : la **volatilité implicite** $$\sigma_{\text{impl}}(K,T)$$

Souvent, on regarde aussi la **moneyness** :
$$
\text{Moneyness}=\frac{K}{S_0}
$$

- $$\frac{K}{S_0}<1$$ : option plutôt **ITM** (call)
- $$\frac{K}{S_0}\approx 1$$ : **ATM**
- $$\frac{K}{S_0}>1$$ : plutôt **OTM**

---

### Smile / Skew (intuition)
- Si $$\sigma_{\text{impl}}$$ varie avec $$K$$ à $$T$$ fixé → on observe un **smile** ou un **skew**
- Si $$\sigma_{\text{impl}}$$ varie avec $$T$$ → structure par terme de volatilité

Une surface complètement “plate” ($$\sigma_{\text{impl}}(K,T)\approx \sigma$$ constante) correspondrait à l’hypothèse
Black–Scholes “idéale”, ce qui est rare sur des données réelles.

---

### Comment on calcule $$\sigma_{\text{impl}}$$ par la méthode de Newton–Raphson ?
On met à jour :
$$
\sigma_{n+1}=\sigma_n-\frac{f(\sigma_n)}{f'(\sigma_n)}
$$
avec
$$
f(\sigma)=C_{\text{BS}}(\sigma)-C_{\text{mkt}}
\qquad\text{et}\qquad
f'(\sigma)=\frac{\partial C_{\text{BS}}}{\partial \sigma}=\text{Vega}
$$

Donc :
$$
\sigma_{n+1}=\sigma_n-\frac{C_{\text{BS}}(\sigma_n)-C_{\text{mkt}}}{\text{Vega}(\sigma_n)}
$$

**Attention :** si $$\text{Vega}$$ est très faible (options très ITM/OTM ou $$T$$ très petit), Newton peut échouer.

""")

    st.divider()

    # =========================
    # INTERFACE
    # =========================
    ticker_label = st.selectbox("Ticker de l'Actif Sous-jacent", EQUITY_TICKERS[9:])
    ticker = ticker_label.split(" ")[0]

    if st.button("Calculer la Surface"):

        # 1. Instanciation des services
        loader = MarketDataLoader()
        model = BlackScholesModel()

        # 2. Barre de progression Streamlit
        progress_bar = st.progress(0)

        # Fonction de callback pour mettre à jour la barre depuis la classe
        def update_progress(p):
            progress_bar.progress(p)

        with st.spinner("Construction de la surface en cours..."):
            # 3. Appel de la méthode statique intégrée
            df_surface = ImpliedVolatility.get_surface(
                ticker=ticker,
                loader=loader,
                model=model,
                max_strikes=40,  # Ajustable selon la vitesse désirée
                progress_callback=update_progress
            )

        progress_bar.empty()

        if not df_surface.empty:
            st.success(f"Surface générée avec {len(df_surface)} points.")

            with st.expander("Voir les données (extrait)"):
                st.dataframe(
                    df_surface.sort_values(["Maturity", "Strike"]).head(30),
                    use_container_width=True
                )

            # 4. Affichage Graphique (Plotly)
            fig = go.Figure(data=[go.Mesh3d(
                x=df_surface["Strike"],
                y=df_surface["Maturity"],
                z=df_surface["Implied Volatility"],
                intensity=df_surface["Implied Volatility"],
                colorscale="Jet",
                opacity=0.8
            )])

            fig.update_layout(
                title=f"Surface de Volatilité Implicite - {ticker}",
                scene=dict(
                    xaxis_title="Strike ($)",
                    yaxis_title="Maturité (Années)",
                    zaxis_title="Volatilité"
                ),
                width=800,
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("Impossible de récupérer suffisamment de données pour tracer la surface.")


def page_structured_products():
    st.header("Structuration de Produits")

    # --- 1. SESSION STATE (Mémoire de la page) ---
    if 'struct_legs' not in st.session_state:
        st.session_state.struct_legs = []
    if "last_structured_product" not in st.session_state:
        st.session_state.last_structured_product = None

    # ----------------------------
    # Helpers (local, safe)
    # ----------------------------
    class UnderlyingLeg:
        """Proxy pour représenter une position sur le sous-jacent: prix ~ spot aujourd'hui, payoff = S_T."""
        def __init__(self, spot0: float):
            self.spot0 = float(spot0)

        def pricing_future(self) -> float:
            # On réutilise la branche pricing_future du StructuredProduct pour le pricer "comme spot"
            return self.spot0

        def payoff(self, spot_at_maturity: float) -> float:
            return float(spot_at_maturity)

    def ensure_option_payoff(opt_obj):
        """
        Si OptionVanille n'a pas payoff(S), on l'ajoute à la volée.
        Ça évite que la structuration casse au moment de tracer le payoff global.
        """
        if hasattr(opt_obj, "payoff"):
            return opt_obj

        # Strike
        K = None
        if hasattr(opt_obj, "parameters"):
            if hasattr(opt_obj.parameters, "strike"):
                K = float(opt_obj.parameters.strike)
            elif hasattr(opt_obj.parameters, "get_strike"):
                K = float(opt_obj.parameters.get_strike())

        if K is None:
            raise AttributeError("OptionVanille n'a pas payoff() et je ne trouve pas parameters.strike.")

        # Type option
        opt_type = None
        if hasattr(opt_obj, "option"):
            opt_type = str(opt_obj.option).lower()
        elif hasattr(opt_obj, "option_type"):
            opt_type = str(opt_obj.option_type).lower()

        if opt_type not in ("call", "put"):
            raise AttributeError("Impossible d'inférer le type d'option (call/put) pour créer payoff().")

        def _payoff(S):
            S = float(S)
            if opt_type == "call":
                return max(S - K, 0.0)
            return max(K - S, 0.0)

        opt_obj.payoff = _payoff
        return opt_obj

    def build_option_leg(spot_ref, maturity, r, K, vol, opt_type):
        params = Parametres(maturity=maturity, risk_free_rate=r, strike=K, volatility=vol)
        inst = OptionVanille(parameters=params, spot=spot_ref, option=opt_type)
        inst = ensure_option_payoff(inst)
        model = BlackScholesModel()
        return inst, model

    def push_strategy_to_builder(strategy_name: str, legs_spec: list):
        """
        legs_spec: list of dicts like:
        {"type":"Option Vanille","quantity":+1,"params":{"strike":100,"type":"call","vol":0.2}}
        """
        for spec in legs_spec:
            st.session_state.struct_legs.append({
                "type": spec["type"],
                "quantity": float(spec["quantity"]),
                "params": spec.get("params", {}),
                "id": len(st.session_state.struct_legs) + 1
            })
        st.success(f"✅ Stratégie ajoutée au builder : {strategy_name}")
        st.rerun()

    # --- 2. PARTIE PÉDAGOGIQUE ---
    with st.expander("📘 Cours : Legs, prix, payoff et PnL (produits structurés)", expanded=False):
        st.markdown("""
    Un produit structuré se comprend comme un assemblage de **jambes (legs)**.
    Chaque jambe est définie par :
    - un **instrument** (option, obligation, sous-jacent, …),
    - une **quantité** $$q_i$$ (positive = *long*, négative = *short*),
    - éventuellement un **modèle** de pricing (BS, arbre, MC…).

    L’idée centrale : **linéarité** (on additionne prix / payoff jambe par jambe).
    """)

        st.latex(r"V_T^{\text{produit}}(S_T) \;=\; \sum_{i=1}^{N} q_i \, V_T^{(i)}(S_T)")
        st.latex(r"P_0^{\text{produit}} \;=\; \sum_{i=1}^{N} q_i \, P_0^{(i)}")

        st.markdown("""
    ### Payoff vs PnL (à maturité)
    - **Payoff** : valeur finale du produit à \(T\)
    - **PnL** : payoff moins le coût initial “porté” jusqu’à \(T\)
    """)
        st.latex(r"\text{PnL}_T(S_T) \;=\; V_T^{\text{produit}}(S_T)\;-\;P_0^{\text{produit}}\,e^{rT}")

        st.markdown("""
    ### Payoffs des briques utilisées ici
    - **Sous-jacent (spot)** (proxy) :
    """)
        st.latex(r"V_T^{\text{underlying}}(S_T) = S_T")

        st.markdown("- **Call européen** :")
        st.latex(r"V_T^{\text{call}}(S_T)=\max(S_T-K,0)")

        st.markdown("- **Put européen** :")
        st.latex(r"V_T^{\text{put}}(S_T)=\max(K-S_T,0)")


    with st.expander("🧠 Hypothèses et limites", expanded=False):
        st.markdown("""
    ### Hypothèses de modélisation (options vanilles)

    Les options vanilles sont évaluées dans un cadre de type **Black–Scholes**, ce qui implique notamment :
    - options **européennes** (exercice à maturité) ;
    - volatilité **constante** sur l’horizon considéré ;
    - taux sans risque **constant** (ou assimilé à un taux moyen sur la maturité) ;
    - marchés frictionless (pas de coûts de transaction, pas de contraintes de liquidité) ;
    - dynamique log-normale du sous-jacent.

    Équation de diffusion (forme standard) :
    """)
        st.latex(r"\frac{dS_t}{S_t}=(r-q)\,dt+\sigma\,dW_t")

        st.markdown("""
    ### Hypothèses de construction du produit (legs)

    Dans cette page, un produit structuré est traité comme une **combinaison linéaire** de jambes (legs) :
    """)
        st.latex(r"P_0^{\text{produit}}=\sum_{i=1}^{N} q_i\,P_0^{(i)} \qquad\text{et}\qquad V_T^{\text{produit}}(S_T)=\sum_{i=1}^{N} q_i\,V_T^{(i)}(S_T)")

        st.markdown("""
    Hypothèses pratiques retenues ici :
    - toutes les jambes partagent une **maturité globale** (même horizon \(T\)) ;
    - une volatilité “globale” peut être utilisée par défaut (modifiable jambe par jambe dans le builder) ;
    - la jambe **Sous-jacent (Spot)** est un proxy pédagogique.
    """)
        st.markdown("Proxy sous-jacent (lecture simple) :")
        st.latex(r"P_0^{\text{underlying}}\approx S_0 \qquad\text{et}\qquad V_T^{\text{underlying}}(S_T)=S_T")

        st.markdown("""
    ### Pistes d’amélioration (extensions naturelles)
    - maturités différentes selon les jambes ;
    - filtres de liquidité (le volume par exemple) ;
    - modèles plus riches (local vol, Heston) et produits avec exercice anticipé (arbres/LSM).
    """)


    with st.expander("🧩 Exemples de stratégies : comment ça marche en legs (avec formules)", expanded=False):
        st.markdown("""
    ### 1) Long Straddle (même strike, même maturité)
    **Legs :**
    - \(+1\) Call(\(K\))
    - \(+1\) Put(\(K\))

    Payoff :
    """)
        st.latex(r"V_T(S_T) = \max(S_T-K,0) + \max(K-S_T,0) = |S_T-K|")

        st.markdown("""
    Coût initial :
    """)
        st.latex(r"P_0 = C_0(K,T) + P_0(K,T)")

        st.markdown("""
    Break-even en **PnL** :
    """)
        st.latex(r"|S_T-K| = (C_0+P_0)\,e^{rT}")

        st.markdown("""
    **Intuition :**
    - tu paies une prime (les 2 options),
    - tu gagnes si le sous-jacent bouge beaucoup (vers le haut ou vers le bas),
    - tu perds si le sous-jacent reste proche de $$K$$.

    ---

    ### 2) Bull Call Spread
    Legs :
    - $$+1$$ Call($$K_1$$)
    - $$-1$$ Call($$K_2$$) avec $$K_1<K_2$$

    Intuition :
    - tu finances une partie du call acheté en vendant un call plus haut,
    - gain plafonné, coût réduit.

    ---

    ### 3) Butterfly (calls)
    Legs :
    - $$+1$$ Call($$K_1$$)
    - $$-2$$ Call($$K_2$$)
    - $$+1$$ Call($$K_3$$) avec $$K_1<K_2<K_3$$

    Intuition :
    - payoff max autour de $$K_2$$,
    - stratégie “range” (tu gagnes si $$S_T$$ reste près du milieu).
    """)


    st.divider()

    # --- 3. PARAMÈTRES GLOBAUX DU PRODUIT ---
    st.subheader("1. Configuration du Marché")

    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    with col_g1:
        global_ticker = st.selectbox("Actif de Référence", EQUITY_TICKERS, key="struct_ticker")
        ticker_symbol = global_ticker.split(" ")[0]
        try:
            spot_ref = DATA_LOADER.get_equity_spot(ticker_symbol)
        except Exception:
            spot_ref = 100.0
        st.metric("Spot Actuel", f"{spot_ref:.2f} €")

    with col_g2:
        global_maturity = st.number_input("Maturité Globale (Années)", 0.05, 10.0, 1.0, 0.1, key="struct_mat")

    with col_g3:
        try:
            r_market = rf_from_curve(global_maturity)
        except Exception:
            r_market = 0.03
        r_input = st.number_input("Taux sans risque (r)", value=float(r_market), format="%.4f", key="struct_rate")
        st.caption(f"Taux marché suggéré : {r_market*100:.2f}%")

    with col_g4:
        vol_global = st.number_input("Volatilité (σ) pour options", value=0.20, format="%.4f")

    st.divider()

    # --- 4. Deux modes : stratégies (pédagogique) + builder (manuel) ---
    tabA, tabB = st.tabs(["🎛️ Stratégies (et comment les construire en legs)", "🧱 Builder (legs sur mesure)"])

    # =========================================================
    # TAB A — Stratégies prédéfinies + explication legs + bouton pour “importer” dans builder
    # =========================================================
    with tabA:
        st.subheader("2. Stratégies prédéfinies (et legs correspondants)")

        strategy = st.selectbox(
            "Choisir une stratégie",
            ["Long Straddle", "Butterfly (Calls)", "Bull Call Spread", "Bear Put Spread"],
            index=0
        )

        validate_vs_spot = st.checkbox("Valider les contraintes de strike", value=True)
        notional = st.number_input("Multiplicateur (notional)", value=1.0, step=0.5)

        # Inputs strikes
        K = K1 = K2 = K3 = None
        if strategy == "Long Straddle":
            K = st.number_input("Strike K", value=float(spot_ref), step=1.0)
            legs_spec = [
                {"type": "Option Vanille", "quantity": +1 * notional, "params": {"strike": float(K), "type": "call", "vol": float(vol_global)}},
                {"type": "Option Vanille", "quantity": +1 * notional, "params": {"strike": float(K), "type": "put",  "vol": float(vol_global)}},
            ]
            st.info("Legs: +1 Call(K) +1 Put(K)")

        elif strategy == "Butterfly (Calls)":
            c1, c2, c3 = st.columns(3)
            with c1:
                K1 = st.number_input("K1", value=float(spot_ref * 0.90), step=1.0)
            with c2:
                K2 = st.number_input("K2", value=float(spot_ref * 1.00), step=1.0)
            with c3:
                K3 = st.number_input("K3", value=float(spot_ref * 1.10), step=1.0)

            legs_spec = [
                {"type": "Option Vanille", "quantity": +1 * notional, "params": {"strike": float(K1), "type": "call", "vol": float(vol_global)}},
                {"type": "Option Vanille", "quantity": -2 * notional, "params": {"strike": float(K2), "type": "call", "vol": float(vol_global)}},
                {"type": "Option Vanille", "quantity": +1 * notional, "params": {"strike": float(K3), "type": "call", "vol": float(vol_global)}},
            ]
            st.info("Legs: +1 Call(K1) −2 Call(K2) +1 Call(K3)")

        elif strategy == "Bull Call Spread":
            c1, c2 = st.columns(2)
            with c1:
                K1 = st.number_input("K1 (achat call)", value=float(spot_ref * 0.95), step=1.0)
            with c2:
                K2 = st.number_input("K2 (vente call)", value=float(spot_ref * 1.05), step=1.0)

            legs_spec = [
                {"type": "Option Vanille", "quantity": +1 * notional, "params": {"strike": float(K1), "type": "call", "vol": float(vol_global)}},
                {"type": "Option Vanille", "quantity": -1 * notional, "params": {"strike": float(K2), "type": "call", "vol": float(vol_global)}},
            ]
            st.info("Legs: +1 Call(K1) −1 Call(K2)")

        else:  # Bear Put Spread
            c1, c2 = st.columns(2)
            with c1:
                K1 = st.number_input("K1 (achat put)", value=float(spot_ref * 1.05), step=1.0)
            with c2:
                K2 = st.number_input("K2 (vente put)", value=float(spot_ref * 0.95), step=1.0)

            legs_spec = [
                {"type": "Option Vanille", "quantity": +1 * notional, "params": {"strike": float(K1), "type": "put", "vol": float(vol_global)}},
                {"type": "Option Vanille", "quantity": -1 * notional, "params": {"strike": float(K2), "type": "put", "vol": float(vol_global)}},
            ]
            st.info("Legs: +1 Put(K1) −1 Put(K2)")

        # validations (soft)
        try:
            if validate_vs_spot:
                if strategy == "Bull Call Spread" and not (K1 < spot_ref < K2):
                    st.warning("⚠️ Pour un Bull Call Spread “classique”, on veut K1 < Spot < K2.")
                if strategy == "Bear Put Spread" and not (K2 < spot_ref < K1):
                    st.warning("⚠️ Pour un Bear Put Spread “classique”, on veut K2 < Spot < K1.")
                if strategy == "Butterfly (Calls)" and not (K1 < K2 < K3):
                    st.warning("⚠️ Butterfly standard: K1 < K2 < K3.")
        except Exception:
            pass

        st.divider()
        st.subheader("3. Construire et analyser (auto)")

        col_btn1, col_btn2 = st.columns([1, 2])
        with col_btn1:
            build_and_price = st.button("🚀 Construire & pricer", type="primary")
        with col_btn2:
            st.caption("Tu peux aussi importer les legs dans le builder pour les modifier à la main (quantités, vol, strikes…).")

        if build_and_price:
            try:
                prod = StructuredProduct(name=f"{strategy} ({ticker_symbol})")

                # Build legs
                for spec in legs_spec:
                    qty = float(spec["quantity"])
                    p = spec["params"]
                    inst, model = build_option_leg(
                        spot_ref=spot_ref,
                        maturity=global_maturity,
                        r=r_input,
                        K=float(p["strike"]),
                        vol=float(p["vol"]),
                        opt_type=str(p["type"]).lower()
                    )
                    prod.add_leg(inst, qty, model, label=f"{p['type'].upper()} K={p['strike']}")

                st.session_state.last_structured_product = prod
                st.success("✅ Produit construit.")

            except Exception as e:
                st.error(f"Erreur construction/pricing : {e}")

        # Button to import into builder (so they see “constructed by hand”)
        if st.button("➕ Importer ces legs dans le Builder", key="import_strategy"):
            push_strategy_to_builder(strategy, legs_spec)

        # Display results if available
        if st.session_state.last_structured_product is not None:
            prod = st.session_state.last_structured_product

            total_price = prod.price()
            greeks = prod.compute_greeks()

            c1, c2, c3 = st.columns(3)
            c1.metric("Prix total (t0)", f"{total_price:,.2f} €")
            c2.metric("Delta global", f"{greeks.get('Delta', 0.0):.4f}")
            c3.metric("Vega global", f"{greeks.get('Vega', 0.0):.4f}")

            st.subheader("Payoff / PnL à maturité")
            show_pnl = st.checkbox("Afficher PnL (payoff - coût capitalisé)", value=True, key="pnl_preset")

            spot_range = np.linspace(spot_ref * 0.5, spot_ref * 1.5, 150)
            df_payoff = prod.generate_payoff_profile(spot_range)

            if show_pnl:
                df_payoff["PnL"] = df_payoff["Payoff Produit"] - (total_price * np.exp(r_input * global_maturity))
                y_col = "PnL"
                ref_y = 0.0
                ref_label = "Break-even (PnL=0)"
            else:
                y_col = "Payoff Produit"
                ref_y = total_price
                ref_label = "Coût initial (référence)"

            base = alt.Chart(df_payoff).encode(x=alt.X('Spot', title='Sous-jacent à maturité'))
            line = base.mark_line(strokeWidth=3).encode(
                y=alt.Y(y_col, title=y_col),
                tooltip=['Spot', alt.Tooltip(y_col, format=',.2f')]
            )
            rule = alt.Chart(pd.DataFrame({'y': [ref_y]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y')
            st.altair_chart((line + rule).properties(height=350), use_container_width=True)
            st.caption(ref_label)

    # =========================================================
    # TAB B — Builder manual (your original idea, but robust)
    # =========================================================
    with tabB:
        # --- Templates rapides dans le Builder ---
        with st.container(border=True):
            st.markdown("### Templates rapides (ajout automatique de legs)")
            cT1, cT2, cT3 = st.columns([1.4, 1, 1])

            with cT1:
                template = st.selectbox(
                    "Template",
                    ["Long Straddle (Call+Put même K)", "Short Straddle (vente Call+Put)"],
                    index=0,
                    key="tpl_struct"
                )

            with cT2:
                K_tpl = st.number_input("Strike K", value=float(spot_ref), step=1.0, key="tpl_K")

            with cT3:
                qty_tpl = st.number_input("Notional", value=1.0, step=0.5, key="tpl_qty")

            st.caption("Ajoute automatiquement les jambes dans la liste ci-dessous (modifiable ensuite).")

            if st.button("➕ Ajouter le template au produit", key="tpl_add"):
                sgn = +1.0 if template.startswith("Long") else -1.0
                st.session_state.struct_legs.append({
                    "type": "Option Vanille",
                    "quantity": sgn * float(qty_tpl),
                    "params": {"strike": float(K_tpl), "type": "call", "vol": float(vol_global)},
                    "id": len(st.session_state.struct_legs) + 1
                })
                st.session_state.struct_legs.append({
                    "type": "Option Vanille",
                    "quantity": sgn * float(qty_tpl),
                    "params": {"strike": float(K_tpl), "type": "put", "vol": float(vol_global)},
                    "id": len(st.session_state.struct_legs) + 1
                })
                st.success("✅ Straddle ajouté au Builder (2 legs).")
                st.rerun()

        st.subheader("2. Ajouter des Composants (Legs)")

        with st.container(border=True):
            c_add1, c_add2, c_add3, c_add4 = st.columns([1.5, 1, 1, 1])

            with c_add1:
                instrument_type = st.selectbox("Type d'Instrument", ["Obligation (Zéro-Coupon)", "Option Vanille", "Sous-jacent (Spot)"])

            with c_add2:
                quantity = st.number_input("Quantité (Poids)", value=1.0, step=0.1, help="Négatif pour une position Short (Vente)")

            params_dict = {}
            with c_add3:
                if instrument_type == "Option Vanille":
                    p_strike = st.number_input("Strike", value=float(spot_ref), step=1.0)
                    p_type = st.selectbox("Type", ["call", "put"])
                    params_dict = {"strike": float(p_strike), "type": p_type, "vol": float(vol_global)}
                elif instrument_type == "Obligation (Zéro-Coupon)":
                    p_nominal = st.number_input("Nominal", value=100.0, step=10.0)
                    p_coupon_rate = st.number_input("Coupon (%)", value=0.0, step=0.1) / 100
                    params_dict = {"nominal": float(p_nominal), "coupon_rate": float(p_coupon_rate)}
                elif instrument_type == "Sous-jacent (Spot)":
                    st.caption("Position directe sur l'action : prix≈spot aujourd'hui, payoff=S_T.")
                    params_dict = {}

            with c_add4:
                st.write("")
                st.write("")
                if st.button("➕ Ajouter", type="primary", key="add_leg_manual"):
                    new_leg = {
                        "type": instrument_type,
                        "quantity": float(quantity),
                        "params": params_dict,
                        "id": len(st.session_state.struct_legs) + 1
                    }
                    st.session_state.struct_legs.append(new_leg)
                    st.rerun()

        st.divider()
        st.subheader("3. Composition du Produit")

        if len(st.session_state.struct_legs) == 0:
            st.info("Votre produit est vide. Ajoutez des composants ci-dessus.")
        else:
            for idx, leg in enumerate(st.session_state.struct_legs):
                with st.expander(f"🧩 Jambe #{idx+1} : {leg['quantity']} x {leg['type']}", expanded=True):
                    c_l1, c_l2 = st.columns([4, 1])
                    with c_l1:
                        if leg['type'] == "Option Vanille":
                            st.write(f"**{leg['params']['type'].upper()}** | Strike: **{leg['params']['strike']}** | Vol: **{leg['params']['vol']:.2%}**")
                        elif leg['type'] == "Obligation (Zéro-Coupon)":
                            st.write(f"Nominal: **{leg['params']['nominal']} €** | Coupon rate: **{leg['params']['coupon_rate']:.2%}**")
                        elif leg['type'] == "Sous-jacent (Spot)":
                            st.write(f"Sous-jacent spot (S0={spot_ref:.2f} €)")

                    with c_l2:
                        if st.button("Supprimer", key=f"del_{idx}"):
                            st.session_state.struct_legs.pop(idx)
                            st.rerun()

        st.divider()
        st.subheader("4. Analyse & Payoff")

        if st.button("Lancer la Structuration", disabled=(len(st.session_state.struct_legs) == 0), key="run_manual"):

            try:
                my_product = StructuredProduct(name="Mon Produit Perso")

                for leg in st.session_state.struct_legs:
                    qty = float(leg['quantity'])
                    l_type = leg['type']
                    p = leg['params']

                    if l_type == "Option Vanille":
                        inst, model = build_option_leg(
                            spot_ref=spot_ref,
                            maturity=global_maturity,
                            r=r_input,
                            K=float(p['strike']),
                            vol=float(p['vol']),
                            opt_type=str(p['type']).lower()
                        )
                        my_product.add_leg(inst, qty, model, label="Option")

                    elif l_type == "Obligation (Zéro-Coupon)":
                        bond_params = Parametres(
                            maturity=global_maturity,
                            risk_free_rate=r_input,
                            strike=0.0,
                            volatility=0.0,
                            dividend=0.0
                        )

                        bond_params.nominal = float(p['nominal'])
                        # coupon = cash coupon, pas le taux
                        bond_params.coupon = float(p['nominal']) * float(p['coupon_rate'])

                        if not hasattr(bond_params, 'get_nominal'):
                            bond_params.get_nominal = lambda: bond_params.nominal
                        if not hasattr(bond_params, 'get_coupon'):
                            bond_params.get_coupon = lambda: bond_params.coupon

                        inst = Bonds(parameters=bond_params)
                        my_product.add_leg(inst, qty, model=None, label="Bond")

                    elif l_type == "Sous-jacent (Spot)":
                        inst = UnderlyingLeg(spot_ref)
                        my_product.add_leg(inst, qty, model=None, label="Underlying")

                total_price = my_product.price()
                greeks = my_product.compute_greeks()

                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Prix Total du Produit", f"{total_price:,.2f} €")
                col_res2.metric("Delta Global", f"{greeks.get('Delta', 0.0):.4f}")
                col_res3.metric("Vega Global", f"{greeks.get('Vega', 0.0):.4f}")

                st.subheader("Profil de Gain à Maturité (Payoff)")
                spot_range = np.linspace(spot_ref * 0.5, spot_ref * 1.5, 150)
                df_payoff = my_product.generate_payoff_profile(spot_range)

                chart = alt.Chart(df_payoff).mark_line(strokeWidth=3).encode(
                    x=alt.X('Spot', title='Prix du Sous-jacent à Maturité'),
                    y=alt.Y('Payoff Produit', title='Valeur Finale du Produit'),
                    tooltip=['Spot', alt.Tooltip('Payoff Produit', format=',.2f')]
                ).properties(height=350)

                st.altair_chart(chart, use_container_width=True)
                st.success("Calcul terminé.")

            except Exception as e:
                st.error(f"Une erreur est survenue lors du pricing : {e}")
                st.info("Astuce : les futures n'ont pas toujours payoff(). Utilise plutôt 'Sous-jacent (Spot)' pour une jambe directionnelle.")


# =================================================================
# --- Fin de l'Application Streamlit ---
# =================================================================
# Routing
if page == "Pricer":
    page_pricer()
elif page == "Produits Structurés":
    page_structured_products()
elif page == "Calibration de la volatilité":
    page_calibration()
elif page == "Surface de vol":
    page_surface()