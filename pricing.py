from scipy.stats import norm
import numpy as np

# --- INIT --- #
if __name__ == "__main__":
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1

def d_1(S0, K, r, sigma, T):
    return (np.log(S0/K)+(r + 1/2 * sigma**2)*T) / ( sigma * np.sqrt(T))

def d_2(d1, sigma, T):
    return d1 - sigma * np.sqrt(T)

def bs_call_price(S0, K, r, sigma, T):
    if T == 0 or sigma == 0:
        return max(S0-K, 0)
    else:
        return S0 * norm.cdf(d_1(S0, K, r, sigma, T)) - K * np.exp(-r*T) * norm.cdf(d_2(d_1(S0, K, r, sigma, T), sigma, T))

def bs_put_price(S0, K, r, sigma, T):
    if T == 0 or sigma == 0:
        return max(K-S0, 0)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d_2(d_1(S0, K, r, sigma, T), sigma, T)) - S0 * norm.cdf(-d_1(S0, K, r, sigma, T))

if __name__ == "__main__":
    C = bs_call_price(S0, K, r, sigma, T)   # ← also unindented
    P = bs_put_price(S0, K, r, sigma, T)

    print(f"Call: {C:.4f}")
    print(f"Put: {P:.4f}")

    lhs = C - P
    rhs = S0 - K * np.exp(-r * T)
    print(f"Parity residual: {lhs - rhs:.2e}")
    assert abs(lhs - rhs) < 1e-10, "Put-call parity failed"

"""
EXPLANATION:
    For the asset pricer you need to see day to day changes. 
        But for options, the only thing that determines the payoff 
        is where the price ends. Because it is a random walk with 
        normally distributed steps, adding up the variables just 
        gives you another normal with summed variances so you can 
        jump straight to the end price by just using a larger 
        variance 
"""

def terminal_prices(S0, r, sigma, T, n_paths, rng):
    Z = rng.normal(size=n_paths)
    S_T = S0 * np.exp((r-0.5*sigma**2)*T + sigma * np.sqrt(T) * Z)
    return S_T

def mc_call_price(S0, K, r, sigma, T, n_paths, rng):
    payoffs = np.maximum(terminal_prices(S0, r, sigma, T, n_paths, rng) - K, 0)
    price = np.exp(-r*T) * np.mean(payoffs)
    se = np.exp(-r*T) * np.std(payoffs, ddof=1) / np.sqrt(n_paths)
    return price, se

