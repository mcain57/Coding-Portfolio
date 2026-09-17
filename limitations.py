import numpy as np
import matplotlib.pyplot as plt
from pricing import bs_call_price, mc_call_price, terminal_prices
import pandas as pd

S0 = 100.0
r = 0.05
sigma = 0.2
T = 1.0
Ns = np.array([10**3, 10**4, 10**5, 10**6])
N_REPEATS = 20
K_array = [100, 120, 150, 200, 250]
n_paths = 1000000
results = {"K": [], "BS": [], "MC": [], "SE": [], "rel SE %": [], "% zero payoff": [], "useful paths": []}

for j in K_array:
    price, se = mc_call_price(S0, j, r, sigma, T, int(n_paths))
    bs = bs_call_price(S0, j, r, sigma, T)
    rel_se = se / price
    S_T = terminal_prices(S0, r, sigma, T, n_paths)
    frac_zero = (S_T <= j).mean()
    n_useful = (S_T > j).sum()

    results["K"].append(j)
    results["BS"].append(bs)
    results["MC"].append(price)
    results["SE"].append(se)
    results["rel SE %"].append(rel_se)
    results["% zero payoff"].append(frac_zero)
    results["useful paths"].append(n_useful)

DF = pd.dataframe(results)
print(DF)

"""
Result: Print a table with columns: K, BS, MC, SE, rel SE %, % zero payoff, useful paths
"""

