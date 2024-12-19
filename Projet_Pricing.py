from tkinter import N
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.stats import norm


#Option Européenne

###########################
#Partie Black & Scholes

def d_j(j, S_0, K, r, sigma, T):
    return (np.log(S_0/K) + (r + ((-1)**(j-1))*0.5*sigma*sigma)*T)/(sigma*(T**0.5))

def vanilla_call_price(S_0, K, r, sigma, T):
    return S_0 * norm.cdf(d_j(1, S_0, K, r, sigma, T)) - K * np.exp(-r * T) * norm.cdf(d_j(2, S_0, K, r, sigma, T))

def vanilla_put_price(S_0, K, r, sigma, T):
    return -S_0 * norm.cdf(-d_j(1, S_0, K, r, sigma, T)) + K * np.exp(-r * T) * norm.cdf(-d_j(2, S_0, K, r, sigma, T))

###########################
#Partie Monte Carlo


#Discretisation Temps
def generate_t(T, num_steps):
    return np.linspace(0, T, num_steps)

def simulate_gbm(S_0, sigma, mu, T, num_steps):
    t = generate_t(T, num_steps)
    S=[S_0]
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        S.append(S[-1]*np.exp((mu-(sigma**2)/2)*dt + sigma*np.sqrt(dt)*np.random.normal(0,1)))
    return S

def process(S, T, mu, sigma, steps, N):
    dt = T/steps
    ST = np.log(S) +  np.cumsum(((mu - sigma**2/2)*dt + sigma*np.sqrt(dt) * np.random.normal(size=(steps,N))),axis=0)    
    return np.exp(ST)

def payoff_call(S,K):
    return np.max([S[-1] - K, 0])

def payoff_put(S,K):
    return np.max([K - S[-1], 0])

def simulate_monte_carlo(option_type, S_0, K, sigma, mu, N, T, num_steps):
    t = generate_t(T, num_steps)
    P = []
    for i in range(N):
        S = simulate_gbm(S_0, sigma, mu, T, num_steps)
        if option_type == "call":
            P.append(payoff_call(S, K))
        elif option_type == "put":
            P.append(payoff_put(S, K))
    return P

def convergence_mc(P, ic):
    a = norm.ppf(ic)
    M = []
    ET = []
    b_inf = []
    b_sup = []
    for i in range(len(P)):
        M.append(np.mean(P[:i+1]))
        ET.append(np.std(P[:i+1]))
        b_inf.append(M[-1] - a * ET[-1] / np.sqrt(i))
        b_sup.append(M[-1] + a * ET[-1] / np.sqrt(i))
    return M, b_sup, b_inf





def euro_option_page():
    st.title("Pricer Equity par méthode Monte Carlo & Black & Scholes pour Option Européenne")
    st.markdown(
        """
        On note $S_{t}$ le prix d'un actif fixé au temps $t$. Le modèle de Black et Scholes consiste à dire que le prix de cet actif répond à l'équation différentielle stochastique suivante:
        
        $$
        dS_{t} = S_{t}(\mu dt + \sigma dW_{t})
        $$

        où $\mu$ est un réel (appelé parfois dérive),
        $\sigma$ est un réel positif (appelé volatilité) et $W$ désigne un mouvement brownien standard.

        On suppose que l'on veuille estimer l'espérance du gain perçu par la détention d'une option européenne d'achat (call) de maturité $T$ et de prix d'exercice (strike) $K$. Si le prix au temps $T$, à savoir $S_{T}$, est plus grand que $K$, l'option est exercée et le gain est de $(S_{T}-K)$, sinon le gain est nul. On cherche donc à estimer la quantité $\mathbb {E} [(S_{T}-K)_{+}]$ où $x_{+}:=\max(0,x)$.
        (Idem pour le put où l'option est exercée si $S_{T} < K$, et le gain est alors $K - S_{T}$ si $S_{T} < K$, sinon le gain est nul. On cherche à estimer $\mathbb {E} [(K-S_{T})_{+}]$.)
        Pour estimer cette espérance, on peut avoir recours à une méthode de Monte-Carlo. Pour cela, il faut alors pouvoir générer des variables aléatoires suivant la loi de $S_{t}$. On peut utiliser directement la forme de la solution de l'équation différentielle stochastique, à savoir

        $$
        S_t = S_0 \\exp(\\mu t - \\frac{\\sigma^2}{2}t +\\sigma W_t)\\quad\\forall t\\geq0.
        $$

        Il suffit alors de générer des variables aléatoires iid de même loi que $W_{T}$, qui est simplement une gaussienne centrée de variance $T$. On approxime ensuite l'espérance du gain de l'option par

        $$
        \\frac{1}{n} \\sum_{k=1}^{n}(S_T^k - K)_+
        $$

        Où : $S_T^1$, ..., $S_T^n$ sont $n$ variables iid de même loi que $S_T = S_0 \\exp(\\mu t - \\frac{\\sigma^2}{2}t + \\sigma W_T)$.
        """)

    # Définition des Paramètres
    st.sidebar.header("Paramètres de la Simulation")
    option_type = st.sidebar.radio("Type d'Option", ["call", "put"])
    S_0 = st.sidebar.number_input("Prix actuel de l'action (S0)", min_value=0.0, value=100.0, step=1.0)
    T = st.sidebar.number_input("Temps jusqu'à l'expiration (années)", min_value=0.1, value=2.0, step=0.1)
    sigma = st.sidebar.number_input("Volatilité (sigma)", min_value=0.001, value=0.15, step=0.001)
    mu = st.sidebar.number_input("Moyenne (mu)", min_value=0.0000, value=0.0150, step=0.001)
    r = st.sidebar.number_input("Taux d'intérêt sans risque (r)", min_value=0.00, value=0.01, step=0.001)
    Ic = st.sidebar.number_input("Intervalle de Confiance", min_value=0.0, value=0.9, step=0.01)
    N = st.sidebar.number_input("Nombre de Simulations", min_value=1, value=1000)
    # Generate a unique key based on the option type
    key_suffix = "call" if option_type == "call" else "put"
    if option_type == "call":
        K = st.sidebar.number_input("Prix d'exercice de l'option (K)", key=f"call_{key_suffix}", min_value=0.0, value=1.05, step=1.0)
    elif option_type == "put":
        K = st.sidebar.number_input("Prix d'exercice de l'option (K)", key=f"put_{key_suffix}", min_value=0.0, value=95.0, step=1.0)


    # Bouton pour lancer les simulations
    if st.button("Lancer les Simulations"):
        #CALL
        if option_type == "call":

            # Graphique du Processus de Diffusion
            D = process(S_0, T, mu, sigma, 200, 100)
            fig_bs, ax_bs = plt.subplots(figsize=(10, 6))
            ax_bs.plot(D)
            ax_bs.set_title('Processus de Diffusion')
            ax_bs.set_xlabel('Time Increments dt')
            ax_bs.set_ylabel("Stock Price S")
            st.pyplot(fig_bs)

            # Simulation Black & Scholes
            call_price_bs = vanilla_call_price(S_0, K, r, sigma, T)

            # Affichage des résultats Black & Scholes
            st.subheader("Résultats de la Simulation Black & Scholes")
            st.write(f"Le Prix de l'Option avec Black & Scholes est {call_price_bs}")

            # Simulation Monte Carlo
          # Choisissez un nombre approprié de pas de temps
            #t = generate_t(T, num_steps) # Utiliser la fonction simulate_gbm avec le vecteur de temps correct

            # Simulation Monte Carlo
            P = simulate_monte_carlo("call",S_0, K, sigma, mu, N, T, 200)
            M, b_inf, b_sup = convergence_mc(P, Ic)
            option_price_mc = np.mean(P)
            error = -(b_sup[-1] - b_inf[-1])/2


            st.subheader("Résultats de la Simulation Monte Carlo")
            st.write(f"Le Prix de l'Option est {option_price_mc} avec une erreur de {error} à un intervalle de confiance de 99%")

            # Graphique de convergence Monte Carlo
            fig_mc, ax_mc = plt.subplots(figsize=(10, 6))
            ax_mc.plot(M, label='Moyenne')
            ax_mc.fill_between(range(len(M)), b_inf, b_sup, alpha=0.3, label='Intervalle de Confiance')
            ax_mc.set_title('Monte Carlo Convergence')
            ax_mc.set_xlabel('Nombre de Simulations')
            ax_mc.set_ylabel("Prix de l'Option")
            ax_mc.legend()
            st.pyplot(fig_mc)
            st.subheader("Conclusion")
            st.write("On remarque que le prix donnée par la Formule de Black Scholes se rapproche fortement de celui donnée par la méthode Monte Carlo")

    
        
        elif option_type == "put":

            # Graphique du Processus de Diffusion
            path = process(S_0, T, mu, sigma, 200, 100)
            fig_bs, ax_bs = plt.subplots(figsize=(10, 6))
            ax_bs.plot(path)
            ax_bs.set_title('Processus de Diffusion')
            ax_bs.set_xlabel('Time Increments dt')
            ax_bs.set_ylabel("Stock Price S")
            st.pyplot(fig_bs)

            # Simulation Black & Scholes
            put_price_bs = vanilla_put_price(S_0, K, r, sigma, T)

            # Affichage des résultats Black & Scholes
            st.subheader("Résultats de la Simulation Black & Scholes")
            st.write(f"Le Prix de l'Option avec Black & Scholes est {put_price_bs}")

            # Simulation Monte Carlo
            P = simulate_monte_carlo("put",S_0, K, sigma, mu, N, T, 200)
            M, b_inf, b_sup = convergence_mc(P, Ic)
            option_price_mc = np.mean(P)
            print(P)
            error = -(b_sup[-1] - b_inf[-1])/2

            # Affichage des résultats Monte Carlo
            st.subheader("Résultats de la Simulation Monte Carlo")
            st.write(f"Le Prix de l'Option avec Monte Carlo est {option_price_mc} avec une erreur de {error} avec un Intervalle de Confiance à 99% ")



            # Graphique de convergence Monte Carlo
            fig_mc, ax_mc = plt.subplots(figsize=(10, 6))
            ax_mc.plot(M, label='Moyenne')
            ax_mc.fill_between(range(len(M)), b_inf, b_sup, alpha=0.3, label='Intervalle de Confiance')
            ax_mc.set_title('Monte Carlo Convergence')
            ax_mc.set_xlabel('Nombre de Simulations')
            ax_mc.set_ylabel("Prix de l'Option")
            ax_mc.legend()
            st.pyplot(fig_mc)
            st.subheader("Conclusion")
            st.write("On remarque que le prix donnée par la Formule de Black Scholes se rapproche fortement de celui donnée par la méthode Monte Carlo")



#Option Asiatique

#####PARTIE Monte Carlo
def generate_t_as():
    t_1 = np.arange(0, 61/252, 1/252)
    t_2 = np.arange(2-59/252, 2, 1/252)
    return np.concatenate((t_1, t_2), axis=0)

def simulate_gbm_as(S_0, sigma, mu, t):
    S=[S_0]
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        S.append(S[-1]*np.exp((mu-(sigma**2)/2)*dt + sigma*np.sqrt(dt)*np.random.normal(0,1)))
    return S

def process_as(S, mu, T, sigma, steps, N):
    dt = T/steps
    ST = np.log(S) +  np.cumsum(((mu - sigma**2/2)*dt + sigma*np.sqrt(dt) * np.random.normal(size=(steps,N))),axis=0)    
    return np.exp(ST)


def payoff_as(S,K):
    return np.max([np.mean(S[61:])/np.mean(S[:60]) - K, 0])

def simulate_montecarlo_as(S_0, K, sigma, mu, Ns):
    t = generate_t_as()
    P = []
    for i in range(Ns):
        S = simulate_gbm_as(S_0,sigma,mu,t)
        P.append(payoff_as(S,K))
    return P

from scipy.stats import norm

def convergence_mc_as(P, ic):
    a = norm.ppf(ic)
    M = []
    ET = []
    b_inf = []
    b_sup = []
    error = []
    for i in range(len(P)):
        M.append(np.mean(P[:i + 1]))
        ET.append(np.std(P[:i + 1]))
        b_inf.append(M[-1] - a * ET[-1] / np.sqrt(i))
        b_sup.append(M[-1] + a * ET[-1] / np.sqrt(i))
        error.append((b_sup[-1]-b_inf[-1])/2)
    return M, b_sup, b_inf, error






def asiatique_option_page():
    st.title("Pricer Equity par méthode Monte Carlo pour Option Asiatique")

    st.markdown( """ L'objectif est de calculer la formule du prix d'un call, d'une Option Asiatique. Pour ce faire, nous examinons la variable suivante : 
    $$
    I_T = \\int_{0}^{T} S_t dt
    $$ 
    
Rappelons qu'une approche courante de la méthode de Monte-Carlo consiste à approximer la variable $I_T$ en utilisant une discrétisation en temps. Cette approche implique la subdivision régulière de l'intervalle $[0, T]$ en $N$ parties égales, notées $h$. Les points équidistants $t_k$ sont alors définis tels que $h = \\frac{T}{N}$ et $t_k = kh$ pour $0 \leq k \leq N - 1 $.

L'intégrale $I_T$, interprétée géométriquement comme l'aire sous la courbe de la fonction, peut être approximée à l'aide de sommes de Riemann. Cette approche revient à approximer la courbe par une somme de fonctions constantes, puis à additionner les aires des rectangles formés.

Dans un contexte de dimension $M$, où $M \geq N$, nous fixons $M$ comme un entier suffisamment grand. En effectuant $M$ simulations.

La formule du prix d'un call asiatique de prix d'exercice $K$, fixé à la date d'échéance $T$, est donnée par :
$$
C_T = \\frac{e^{-rT}}{M} \\sum_{i=1}^{M} \\left(\\frac{1}{N}\\sum_{k=1}^{N-1} S^{(i)}_{t_k} - K\\right)_+ 

$$

où $S^{(i)}_{t_k}$ représente le prix de l'actif simulé au temps $t_k$ pour la $i_{ème}$ simulation, et $r$ est le taux d'intérêt. 
Cette formule reflète le payoff de l'option asiatique, prenant en compte la moyenne arithmétique des prix sur la période d'observation.""")


    # ... (Structure for Tunnel option page with Tunnel-specific parameters and calculations)

    # Définition des Paramètres
    st.sidebar.header("Paramètres de la Simulation")
    S_0 = st.sidebar.number_input("Prix actuel de l'action (S0)", min_value=0, value=100, step=100)
    K = st.sidebar.number_input("Prix d'exercice de l'option (K)", min_value=0.0, value=1.05, step=1.0)
    T = st.sidebar.number_input("Temps jusqu'à l'expiration (années)", min_value=0.1, value=2.0, step=0.1)
    sigma = st.sidebar.number_input("Volatilité (sigma)", min_value=0.001, value=0.15, step=0.001)
    mu = st.sidebar.number_input("Moyenne (mu)", min_value=0.0000, value=0.0150, step=0.001)
    r = st.sidebar.number_input("Taux d'intérêt sans risque (r)", min_value=0.00, value=0.01, step=0.001)
    Ic = st.sidebar.number_input("Intervalle de Confiance", min_value=0.0, value=0.90, step=0.01)
    N = st.sidebar.number_input("Nombre de Simulations", min_value=100, value=1000, step=100)

    # Bouton pour lancer les simulations
    if st.button("Lancer les Simulations"):

            # Simulation Monte Carlo
          # Choisissez un nombre approprié de pas de temps
            #t = generate_t(T, num_steps) # Utiliser la fonction simulate_gbm avec le vecteur de temps correct


            # Simulation Monte Carlo
            #N=15000
            P = simulate_montecarlo_as(S_0, K, sigma, mu, N)
            M, b_inf, b_sup, error = convergence_mc_as(P, Ic)
            Price = np.mean(P)
            path= process(S_0, T, mu, sigma, 200, 1000)

            # Graphique du Processus de Diffusion
            fig_bs, ax_bs = plt.subplots(figsize=(10, 6))
            ax_bs.plot(path)
            ax_bs.plot(K)
            ax_bs.set_title('Processus de Diffusion')
            ax_bs.set_xlabel('Time Increments dt')
            ax_bs.set_ylabel("Stock Price S")
            st.pyplot(fig_bs)

            st.subheader("Résultats de la Simulation Monte Carlo")
            st.write(f"Le Prix de l'Option est {Price} avec une erreur de {error[-1]} à un intervalle de confiance de 99%")

            # Graphique de convergence Monte Carlo
            fig_mc, ax_mc = plt.subplots(figsize=(10, 6))
            ax_mc.plot(M, label='Moyenne')
            ax_mc.fill_between(range(len(M)), b_inf, b_sup, alpha=0.3, label='Intervalle de Confiance')
            ax_mc.set_title('Monte Carlo Convergence')
            ax_mc.set_xlabel('Nombre de Simulations')
            ax_mc.set_ylabel("Prix de l'Option")
            ax_mc.legend()
            st.pyplot(fig_mc)


#Option Tunnel
def generate_t_Tunnel():
    return np.arange(0, 2, 1/4)

def simulate_gbm_tunnel(S_0, sigma, mu, t):
    S=[S_0]
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        S.append(S[-1]*np.exp((mu-(sigma**2)/2)*dt + sigma*np.sqrt(dt)*np.random.normal(0,1)))
    return S

def process_tunnel(S, T, mu, sigma, steps, N):
    dt = T/steps
    ST = np.log(S) +  np.cumsum(((mu - sigma**2/2)*dt + sigma*np.sqrt(dt) * np.random.normal(size=(steps,N))),axis=0)    
    return np.exp(ST)

def payoff_Tunnel(S,K):
    payoff = 0
    for i in S[1:len(S)-1]:
        if i > K[0] and i < K[1]:
            payoff += i/S[0]*0.01
        elif i > K[1]:
            payoff += i/S[0]*0.02
        else:
            payoff = 0
    return payoff + np.max([S[-1]/S[0]-K[1],0])

def simulate_montecarlo_Tunnel(S_0, K, sigma, mu, N):
    t = generate_t_Tunnel()
    P = []
    for i in range(N):
        S = simulate_gbm_tunnel(S_0,sigma,mu,t)
        P.append(payoff_Tunnel(S,K))
    return P

from scipy.stats import norm

def convergence_mc_Tunnel(P, ic):
    a = norm.ppf(ic)
    M = []
    ET = []
    b_inf = []
    b_sup = []
    error = []
    for i in range(len(P)):
        M.append(np.mean(P[:i+1]))
        ET.append(np.std(P[:i+1]))
        b_inf.append(M[-1] - a*ET[-1]/np.sqrt(i))
        b_sup.append(M[-1] + a*ET[-1]/np.sqrt(i))
        error.append((b_sup[-1]-b_inf[-1])/2)
    return M, b_sup, b_inf, error



def tunnel_option_page():
    st.title("Pricer Equity par méthode Monte Carlo pour Option Tunnel")
    st.markdown(""" L'option de type tunnel est une variante d'options financières qui introduit des niveaux de barrière spécifiques, influençant le payoff de l'option. Dans le cadre du modèle de Black-Scholes, la dynamique du prix de l'actif sous-jacent $S_t$ est décrite par l'équation différentielle stochastique suivante :

    $$ 
    dS_t = S_t(\\mu dt + \\sigma dW_t)
    $$


    où $S_t$ représente le prix de l'actif, $\\mu$ est la dérive, $\\sigma$ est la volatilité, et $dW_t$ est un incrément du mouvement brownien standard.

    Le payoff d'une option de type tunnel est déterminé en fonction des niveaux de barrière $K$. Supposons que les barrières soient définies comme $K[0]$ et $K[1]$, où $K[0] < K[1]$. Le payoff $P$ est calculé comme suit :""")

    st.latex(r"""
    P = \sum_{i=1}^{T-1} \begin{Bmatrix}
    \frac{S_i}{S_0} \times 0.01 & \text{si } K[0] < S_i < K[1] \\
    \frac{S_i}{S_0} \times 0.02 & \text{si } S_i \geq K[1] \\
    0 & \text{sinon}
    \end{Bmatrix} 
    + \max\left(\frac{S_T}{S_0} - K[1], 0\right)
    """)

    st.markdown(""" où $S_i$ représente le prix de l'actif à l'instant $i$, $S_0$ est le prix initial, et $S_T$ est le prix de l'actif à l'échéance.""")

    # Définition des Paramètres
    st.sidebar.header("Paramètres de la Simulation")
    S_0 = st.sidebar.number_input("Prix actuel de l'action (S0)", min_value=0, value=100, step=10)
    K_1 = st.sidebar.number_input("Prix d'exercice de l'option (K1)", min_value=0.0, value=0.9, step=1.0)
    K_2 = st.sidebar.number_input("Prix d'exercice de l'option (K2)", min_value=0.0, value=1.10, step=1.0)
    T = st.sidebar.number_input("Temps jusqu'à l'expiration (années)", min_value=0.1, value=2.0, step=0.1)
    sigma = st.sidebar.number_input("Volatilité (sigma)", min_value=0.001, value=0.15, step=0.001)
    mu = st.sidebar.number_input("Moyenne (mu)", min_value=0.0000, value=0.0150, step=0.001)
    r = st.sidebar.number_input("Taux d'intérêt sans risque (r)", min_value=0.00, value=0.01, step=0.001)
    Ic = st.sidebar.number_input("Intervalle de Confiance", min_value=0.0, value=0.90, step=0.01)
    N = st.sidebar.number_input("Nombre de Simulations", min_value=100, value=1000, step=100)
    K = [K_1, K_2]

    # Bouton pour lancer les simulations
    if st.button("Lancer les Simulations"):
        
        #N=5000
        P = simulate_montecarlo_Tunnel(S_0, K, sigma, mu, N)
        Price = np.mean(P)
        M, b_inf, b_sup, error = convergence_mc_Tunnel(P, Ic)
        path= process_tunnel(S_0, T, mu, sigma, 200, 1000)

        # Graphique du Processus de Diffusion
        fig_bs, ax_bs = plt.subplots(figsize=(10, 6))
        ax_bs.plot(path)
        
        # Ajouter des annotations texte pour K1 et K2
        ax_bs.annotate(f'K1={K_1}', xy=(0, K_1), xytext=(10, K_1 + 10),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))
        ax_bs.annotate(f'K2={K_2}', xy=(0, K_2), xytext=(10, K_2 + 10),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))
        
        ax_bs.set_title('Processus de Diffusion')
        ax_bs.set_xlabel('Time Increments dt')
        ax_bs.set_ylabel("Stock Price S")
        st.pyplot(fig_bs)


        st.subheader("Résultats de la Simulation Monte Carlo")
        st.write(f"Le Prix de l'Option est {Price} avec une erreur de {error[-1]} à un intervalle de confiance de 99%")

            # Graphique de convergence Monte Carlo
        fig_mc, ax_mc = plt.subplots(figsize=(10, 6))
        ax_mc.plot(M, label='Moyenne')
        ax_mc.fill_between(range(len(M)), b_inf, b_sup, alpha=0.3, label='Intervalle de Confiance')
        ax_mc.set_title('Monte Carlo Convergence')
        ax_mc.set_xlabel('Nombre de Simulations')
        ax_mc.set_ylabel("Prix de l'Option")
        ax_mc.legend()
        st.pyplot(fig_mc)





# Fonction principale de l'application Streamlit
def main():
    pages = ["Option - Européenne", "Option - Tunnel", "Option - Asiatique"]
    selected_page = st.sidebar.radio("Sélectionner l'Option", pages)

    if selected_page == "Option - Européenne":
        euro_option_page()
    elif selected_page == "Option - Tunnel":
        tunnel_option_page()
    elif selected_page == "Option - Asiatique":
        asiatique_option_page()

if __name__ == "__main__":
    main()


