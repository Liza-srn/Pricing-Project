# Option Pricing Calculator

## 📊 Overview

This project is a comprehensive option pricing calculator implemented in Python using Streamlit. It provides pricing calculations for three types of options:

- European Options (using both Black-Scholes and Monte Carlo methods)
- Asian Options (using Monte Carlo method)
- Tunnel Options (using Monte Carlo method)

The application provides an interactive interface for users to input parameters and visualize option pricing results, including price paths and convergence analysis.

## 🚀 Features

- **European Options**:
  - Black-Scholes pricing model
  - Monte Carlo simulation
  - Call and Put options support
  - Price path visualization
  - Convergence analysis

- **Asian Options**:
  - Monte Carlo simulation
  - Average price calculation
  - Confidence interval estimation
  - Interactive visualizations

- **Tunnel Options**:
  - Double barrier implementation
  - Monte Carlo pricing
  - Custom payoff structures
  - Visual price path analysis

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/option-pricing-calculator.git
cd option-pricing-calculator
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```
streamlit
numpy
matplotlib
scipy
```

## 🖥️ Usage

1. Run the Streamlit application:
```bash
streamlit run Projet_Pricing.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Select the type of option you want to price from the sidebar

4. Input the required parameters:
   - Current stock price (S0)
   - Strike price (K)
   - Time to maturity (T)
   - Volatility (σ)
   - Risk-free rate (r)
   - Number of simulations (N)



## 🧮 Mathematical Background

The project implements various mathematical models:

### Black-Scholes Model
For European options, the Black-Scholes formula is used:

$dS_t = S_t(μdt + σdW_t)$


### Option Payoffs and Pricing Formulas

#### European Options
The payoff of European options at maturity T:

For a Call option:
$Payoff_{call} = \max(S_T - K, 0)$

For a Put option:
$Payoff_{put} = \max(K - S_T, 0)$

where $S_T$ is the stock price at maturity and $K$ is the strike price.

Black-Scholes formula for a call option:
$C = S_0N(d_1) - Ke^{-rT}N(d_2)$

where:
$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$
$d_2 = d_1 - \sigma\sqrt{T}$

#### Asian Options
The payoff of an Asian option depends on the average price of the underlying asset:

$Payoff_{asian} = \max(S_{avg} - K, 0)$

where:
$S_{avg} = \frac{1}{N} \sum_{i=1}^N S_i$

The Monte Carlo price is estimated as:

$C_T = \frac{e^{-rT}}{M} \sum_{i=1}^M \max(\frac{1}{N}\sum_{k=1}^N S^{(i)}_{t_k} - K, 0)$


### Monte Carlo Simulation
Used for all option types, with specific implementations for each:
- European: Standard GBM simulation
- Asian: Average price calculation
- Tunnel: Barrier monitoring and custom payoff structure
