import numpy as np
import matplotlib.pyplot as plt
from pricing import bs_call_price, mc_call_price

S0 = 100.0
K = 100.0
r = 0.05
sigma = 0.2
T = 1.0
Ns = np.array([10**3, 10**4, 10**5, 10**6])
N_REPEATS = 20

exact = bs_call_price(S0, K, r, sigma, T)
errors = np.empty(len(Ns))

for i, n in enumerate(Ns):
    reps = [abs(mc_call_price(S0, K, r, sigma, T, int(n),
                              np.random.default_rng(seed))[0] - exact)
            for seed in range(N_REPEATS)]
    errors[i] = np.mean(reps)
    print(f"N = {n:>9,}   mean |error| = {errors[i]:.5f}")
slope, intercept = np.polyfit(np.log(Ns), np.log(errors), 1)

print(f"\nSlope: {slope:+.4f}  (theory -0.5000)")
print(f"Error per 10x paths: {10**slope:.3f}  (theory {10**-0.5:.3f})")

plt.loglog(Ns, errors, "o", label="measured mean |error|")
plt.loglog(Ns, np.exp(intercept) * Ns.astype(float)**slope,
           label=f"fit: slope {slope:+.3f}")
plt.loglog(Ns, errors[0] * (Ns / Ns[0])**-0.5, "--", label="theory: slope -0.5")
plt.xlabel("number of paths, $N$")
plt.ylabel(r"$|C_{MC} - C_{BS}|$")
plt.title(f"Monte Carlo convergence ({N_REPEATS} seeds per $N$)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("convergence.png", dpi=130)
plt.show()
