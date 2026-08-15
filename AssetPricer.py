"""Asset price simulators: GARCH(1,1) volatility and Geometric Brownian Motion."""

"""
Plan: Monte Carlo option pricing     - simulate many paths under risk-neutral drift (r), average the discounted payoff
      Black-Scholes closed form      - code the formula, check it matches the Mont Carlo price
      Real data + implied volatility - pull an option chain, invert Black-Scholes, plot the volatility smile
      Calibrate GARCH to real historical returns
      Calibrate Merton/Heston to the real smile
      Backtest a voltaility signal, report Sharpe

Monte Carlo option pricing:
    Simulate hundreds or thousands and average to find the average payoff
    For European call:
        Simulate many possible end prices, S_T, with GBM
        Payoff off = max(S_T - K, 0)
        Average all the payoffs
        Discount to today: multiply by e^(-rT)

Black-Scholes (different to GBM):
    GBM is a model of how the stock price moves
    Black-Scholes is the formula for an option's price you get by assuming it follows GBM
    C = S_0 * N(d1) - K * e^(-rT) * N(d2)
    where N is the standard normal CDF
    Monte Carlo price should converge to this formula

Real data + implied volatility
    use yfinance, pick a ticker, get the spot price and an option chain for one expiry (ticker.option_chain(date) gives strikes + market prices
    inputs: r = a constant risk-free rate ~4-5%
            T = time to expiery in years
    Per option: root find the implied volatility
    Plot implied volatility against strike giving the volatility smile
"""

import numpy as np

# Default model parameters
ALPHA = 0.10        # GARCH: reaction to last shock
BETA = 0.85         # GARCH: persistence  (keep ALPHA + BETA < 1)
MU = 0.0005         # GBM: drift
DT = 1 / 252        # one step = one trading day


def price_mover(a, shock):
    """Apply one multiplicative price move."""
    return a * (1 + shock)


def asset_garch(S0, n_steps, vol, alpha=ALPHA, beta=BETA, seed=None):
    """Simulate a price path with GARCH(1,1) volatility. Returns (prices, vols)."""
    rng = np.random.default_rng(seed)
    omega = vol**2 * (1 - alpha - beta)
    var = vol**2
    a = S0
    prices = np.empty(n_steps)
    vols = np.empty(n_steps)
    for i in range(n_steps):
        sigma = np.sqrt(var)
        prices[i] = a
        vols[i] = sigma
        shock = rng.normal(0, sigma)
        a = price_mover(a, shock)
        var = omega + alpha * shock**2 + beta * var
    return prices, vols


def asset_gbm(S0, n_steps, vol, mu=MU, dt=DT, seed=None):
    """Simulate a price path with Geometric Brownian Motion (Ito-corrected). Returns prices."""
    rng = np.random.default_rng(seed)
    a = S0
    prices = np.empty(n_steps)
    for i in range(n_steps):
        prices[i] = a
        Z = rng.normal()
        drift = (mu - 0.5 * vol**2) * dt
        shock = vol * np.sqrt(dt) * Z
        a = a * np.exp(drift + shock)
    return prices


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt

    S0 = random.randint(10, 10_000)
    n_steps = int(input("How many price movements should there be: "))
    vol = float(input("Volatility: "))

    prices, vols = asset_garch(S0, n_steps, vol)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(prices); ax1.set_ylabel("Price ($)")
    ax1.set_title(f"Asset A - start ${S0}, GARCH volatility")
    ax2.plot(vols); ax2.set_ylabel("Volatility"); ax2.set_xlabel("Time step")
    plt.tight_layout(); plt.show()

    prices_gbm = asset_gbm(S0, n_steps, vol)
    plt.plot(prices_gbm); plt.ylabel("Price ($)"); plt.xlabel("Time")
    plt.title("Asset pricing using GBM"); plt.show()