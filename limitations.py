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
rng=np.random.default_rng(0)
results = {"K": [], "BS": [], "MC": [], "SE": [], "rel SE": [], "zero payoff": [], "useful paths": []}

for j in K_array:
    price, se = mc_call_price(S0, j, r, sigma, T, int(n_paths), rng)
    bs = bs_call_price(S0, j, r, sigma, T)
    rel_se = se / price
    S_T = terminal_prices(S0, r, sigma, T, n_paths, rng)
    frac_zero = (S_T <= j).mean()
    n_useful = (S_T > j).sum()

    results["K"].append(j)
    results["BS"].append(bs)
    results["MC"].append(price)
    results["SE"].append(se)
    results["rel SE"].append(rel_se)
    results["zero payoff"].append(frac_zero)
    results["useful paths"].append(n_useful)

DF = pd.DataFrame(results)
print(DF)

"""
Result: Print a table with columns: K, BS, MC, SE, rel SE %, % zero payoff, useful paths
"""

"""
Results: 
     K         BS         MC        SE  rel SE %  % zero payoff  useful paths
0  100  10.450584  10.469705  0.014736  0.001407       0.439939        560061
1  120   3.247477   3.235754  0.008650  0.002673       0.777073        222927
2  150   0.359630   0.361392  0.002847  0.007878       0.969646         30354
3  200   0.004799   0.004677  0.000322  0.068869       0.999570           430
4  250   0.000048   0.000039  0.000020  0.519561       0.999996             4
"""