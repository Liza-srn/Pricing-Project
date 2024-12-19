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

## 📸 Screenshots

To add screenshots of your application, save them in an `images` folder and include them in your README like this:

![European Option Interface](images/european_option.png)
*European Option Pricing Interface*

![Asian Option Interface](images/asian_option.png)
*Asian Option Pricing Interface*

![Tunnel Option Interface](images/tunnel_option.png)
*Tunnel Option Pricing Interface*

## 🧮 Mathematical Background

The project implements various mathematical models:

### Black-Scholes Model
For European options, the Black-Scholes formula is used:
```
dS_t = S_t(μdt + σdW_t)
```

### Option Payoffs and Pricing Formulas

#### European Options
The payoff of European options at maturity T:

For a Call option:
```
Payoff = max(S_T - K, 0)
```

For a Put option:
```
Payoff = max(K - S_T, 0)
```

where S_T is the stock price at maturity and K is the strike price.

Black-Scholes formula for a call option:
```
C = S_0N(d1) - Ke^{-rT}N(d2)

where:
d1 = [ln(S_0/K) + (r + σ²/2)T] / (σ√T)
d2 = d1 - σ√T
```

#### Asian Options
The payoff of an Asian option depends on the average price of the underlying asset:
```
Payoff = max(S_avg - K, 0)

where:
S_avg = 1/N ∑ S_i for i from 1 to N
```

The Monte Carlo price is estimated as:
```
C_T = e^{-rT}/M ∑(1/N ∑S^{(i)}_{t_k} - K)_+
for i from 1 to M simulations
```

#### Tunnel Options
The payoff structure for tunnel options combines barrier monitoring with final payoff:
```
P = ∑_{i=1}^{T-1} {
    S_i/S_0 × 0.01  if K[0] < S_i < K[1]
    S_i/S_0 × 0.02  if S_i ≥ K[1]
    0               otherwise
} + max(S_T/S_0 - K[1], 0)
```

where K[0] and K[1] are the lower and upper barriers respectively.

### Monte Carlo Simulation
Used for all option types, with specific implementations for each:
- European: Standard GBM simulation
- Asian: Average price calculation
- Tunnel: Barrier monitoring and custom payoff structure

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any questions or feedback, please reach out to [your-email@example.com]

---
⭐️ If you find this project useful, please consider giving it a star!
