import numpy as np
import matplotlib.pyplot as plt
from pricing import bs_call_price
import pandas as pd

S0 = 100
K = 100
T = 1
sigma = 0.2
r = 0.05

# --- R Sweep --- #

R_results = []
r_vals = np.linspace(0, 0.08, 40)
for i in r_vals:
    bs_result = bs_call_price(S0, K, i, sigma, T)
    R_results.append(bs_result)

# --- Sigma Sweep --- #

Sigma_results = []
sigma_vals = np.linspace(0.1, 0.4, 40)
for j in sigma_vals:
    bs_result = bs_call_price(S0, K, r, j, T)
    Sigma_results.append(bs_result)

# --- Plot Results --- #

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(r_vals, R_results, color="steelblue")
ax1.set_xlabel("risk-free rate $r$")
ax1.set_ylabel("call price")
ax1.set_title("Sensitivity to $r$  ($\\sigma$ = 0.2)")

ax2.plot(sigma_vals, Sigma_results, color="darkorange")
ax2.set_xlabel("volatility $\\sigma$")
ax2.set_ylabel("call price")
ax2.set_title("Sensitivity to $\\sigma$  ($r$ = 0.05)")

for ax in (ax1, ax2):
    ax.grid(alpha=0.3)

lo = min(min(R_results), min(Sigma_results))
hi = max(max(R_results), max(Sigma_results))
ax1.set_ylim(lo - 1, hi + 1)
ax2.set_ylim(lo - 1, hi + 1)

plt.tight_layout()
plt.savefig("sensitivity.png", dpi=130)
plt.show()

r_range = max(R_results) - min(R_results)
s_range = max(Sigma_results) - min(Sigma_results)
print(f"r sweep range:     {r_range:.4f}")
print(f"sigma sweep range: {s_range:.4f}")
print(f"ratio:             {s_range/r_range:.2f}x")