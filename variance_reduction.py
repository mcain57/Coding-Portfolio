"""
Variance reduction: antithetic variates against standard Monte Carlo.
 
Both estimators use the same total number of paths, so the comparison is
like for like. What changes is the precision, not the answer.
 
Run:  python variance_reduction.py
"""
 
import numpy as np
import matplotlib.pyplot as plt
 
from pricing import bs_call_price, mc_call_price, mc_call_price_antithetic
 
S0, R, SIGMA, T = 100.0, 0.05, 0.2, 1.0
N_PATHS = 100_000
STRIKES = [100, 120, 150]
SEED = 0
 
 
def compare(K, n_paths, seed=SEED):
    """Price one option both ways and return the comparison as a dict."""
    exact = bs_call_price(S0, K, R, SIGMA, T)
 
    # Fresh generator for each method so neither gets a luckier stream.
    p_std, se_std = mc_call_price(
        S0, K, R, SIGMA, T, n_paths, np.random.default_rng(seed))
    p_ant, se_ant = mc_call_price_antithetic(
        S0, K, R, SIGMA, T, n_paths, np.random.default_rng(seed))
 
    ratio = se_ant / se_std
    return {
        "K": K,
        "bs": exact,
        "p_std": p_std, "se_std": se_std, "z_std": (p_std - exact) / se_std,
        "p_ant": p_ant, "se_ant": se_ant, "z_ant": (p_ant - exact) / se_ant,
        "ratio": ratio,
        "speedup": 1 / ratio**2,
    }
 
 
if __name__ == "__main__":
    print(f"S0 = {S0}, r = {R}, sigma = {SIGMA}, T = {T}, "
          f"N = {N_PATHS:,} paths\n")
 
    results = [compare(K, N_PATHS) for K in STRIKES]
 
    # ---- table -------------------------------------------------------
    print(f"{'K':>5} {'BS':>10} | {'MC std':>10} {'SE':>9} {'z':>6} "
          f"| {'MC anti':>10} {'SE':>9} {'z':>6} | {'SE ratio':>9} {'speed-up':>9}")
    print("-" * 100)
    for d in results:
        print(f"{d['K']:>5} {d['bs']:>10.4f} | "
              f"{d['p_std']:>10.4f} {d['se_std']:>9.5f} {d['z_std']:>+6.2f} | "
              f"{d['p_ant']:>10.4f} {d['se_ant']:>9.5f} {d['z_ant']:>+6.2f} | "
              f"{d['ratio']:>9.3f} {d['speedup']:>8.2f}x")
 
    atm = results[0]
    print(f"\nAt the money: antithetic sampling cuts the standard error to "
          f"{atm['ratio']*100:.1f}% of the standard estimator,")
    print(f"equivalent to {atm['speedup']:.2f}x as many paths for the same precision.")
    print(f"The benefit falls to {results[-1]['speedup']:.2f}x at K = {results[-1]['K']}, "
          f"where the payoff is less linear in Z.")
 
    # ---- correctness: neither method should be biased ------------------
    for d in results:
        assert abs(d["z_std"]) < 4, f"standard MC off at K={d['K']}"
        assert abs(d["z_ant"]) < 4, f"antithetic MC off at K={d['K']}"
    print("\nBoth estimators agree with the closed form at every strike.")
 
    # ---- convergence comparison ---------------------------------------
    Ns = np.array([10**3, 10**4, 10**5, 10**6])
    exact = bs_call_price(S0, 100, R, SIGMA, T)
    err_std, err_ant = [], []
    for n in Ns:
        e_s = [abs(mc_call_price(S0, 100, R, SIGMA, T, int(n),
                                 np.random.default_rng(s))[0] - exact)
               for s in range(20)]
        e_a = [abs(mc_call_price_antithetic(S0, 100, R, SIGMA, T, int(n),
                                            np.random.default_rng(s))[0] - exact)
               for s in range(20)]
        err_std.append(np.mean(e_s))
        err_ant.append(np.mean(e_a))
 
    plt.figure(figsize=(7, 5))
    plt.loglog(Ns, err_std, "o-", label="standard")
    plt.loglog(Ns, err_ant, "s-", label="antithetic")
    plt.xlabel("total number of paths $N$")
    plt.ylabel(r"mean $|C_{MC} - C_{BS}|$")
    plt.title("Antithetic variates vs standard Monte Carlo (20 seeds per $N$)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("variance_reduction.png", dpi=130)
    print("\nSaved variance_reduction.png")
    plt.show()