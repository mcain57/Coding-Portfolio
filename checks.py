"""
Validation harness for Steps 0-4 of the Monte Carlo / Black-Scholes build.
 
Drop this next to pricing.py and run:  python checks.py
 
It imports YOUR functions and checks them. It doesn't reimplement anything,
so if a check fails, the bug is in pricing.py, not here.
"""
 
import numpy as np
import matplotlib.pyplot as plt
 
from pricing import (
    bs_call_price,
    bs_put_price,
    terminal_prices,
    mc_call_price,
)
 
# Reference case used throughout
S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.2, 1.0
 
passed, failed = [], []
 
 
def check(name, condition, detail=""):
    (passed if condition else failed).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
 
 
# ----------------------------------------------------------------------
print("\n1. Black-Scholes spot check")
 
C = bs_call_price(S0, K, R, SIGMA, T)
P = bs_put_price(S0, K, R, SIGMA, T)
check("call = 10.4506", abs(C - 10.450583572185565) < 1e-9, f"got {C:.10f}")
check("put  =  5.5735", abs(P - 5.573526022256971) < 1e-9, f"got {P:.10f}")
 
# ----------------------------------------------------------------------
print("\n2. Put-call parity, 2000 random parameter sets")
 
rng = np.random.default_rng(0)
worst, worst_params = 0.0, None
for _ in range(2000):
    s0 = rng.uniform(20, 200)
    k = rng.uniform(20, 200)
    r = rng.uniform(0.0, 0.10)
    sig = rng.uniform(0.05, 0.80)
    t = rng.uniform(0.01, 5.0)
    resid = (bs_call_price(s0, k, r, sig, t) - bs_put_price(s0, k, r, sig, t)) - (
        s0 - k * np.exp(-r * t)
    )
    if abs(resid) > worst:
        worst, worst_params = abs(resid), (s0, k, r, sig, t)
check("max |residual| < 1e-9", worst < 1e-9, f"worst {worst:.2e}")
if worst >= 1e-9:
    print(f"        worst at S0={worst_params[0]:.2f} K={worst_params[1]:.2f} "
          f"r={worst_params[2]:.4f} sigma={worst_params[3]:.4f} T={worst_params[4]:.3f}")
 
# ----------------------------------------------------------------------
print("\n3. Functions actually respond to their arguments")
 
# This is the check that catches a d1 computed once at module level.
strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
calls = np.array([bs_call_price(S0, k, R, SIGMA, T) for k in strikes])
puts = np.array([bs_put_price(S0, k, R, SIGMA, T) for k in strikes])
 
check("call price strictly decreasing in K", np.all(np.diff(calls) < 0))
check("put price strictly increasing in K", np.all(np.diff(puts) > 0))
check("all call prices >= 0", np.all(calls >= 0), f"min {calls.min():.4f}")
check("all put prices >= 0", np.all(puts >= 0), f"min {puts.min():.4f}")
 
vols = np.array([bs_call_price(S0, K, R, s, T) for s in [0.1, 0.2, 0.3, 0.4]])
check("call price increasing in sigma", np.all(np.diff(vols) > 0))
 
mats = np.array([bs_call_price(S0, K, R, SIGMA, t) for t in [0.25, 0.5, 1.0, 2.0]])
check("call price increasing in T", np.all(np.diff(mats) > 0))
 
# Deep ITM / OTM limits
deep_itm = bs_call_price(S0, 1.0, R, SIGMA, T)
check("deep ITM call ~ S0 - K*e^(-rT)", abs(deep_itm - (S0 - 1.0 * np.exp(-R * T))) < 1e-6,
      f"got {deep_itm:.6f}")
check("deep OTM call ~ 0", bs_call_price(S0, 10_000.0, R, SIGMA, T) < 1e-6)
 
# ----------------------------------------------------------------------
print("\n4. Edge cases")
 
check("T=0 call = intrinsic", abs(bs_call_price(120.0, 100.0, R, SIGMA, 0.0) - 20.0) < 1e-12)
check("T=0 put  = intrinsic", abs(bs_put_price(80.0, 100.0, R, SIGMA, 0.0) - 20.0) < 1e-12)
check("sigma=0 call = intrinsic", abs(bs_call_price(120.0, 100.0, R, 0.0, T) - 20.0) < 1e-12)
 
# ----------------------------------------------------------------------
print("\n5. terminal_prices distribution")
 
rng = np.random.default_rng(42)
N = 1_000_000
S_T = terminal_prices(S0, R, SIGMA, T, N, rng)
 
check("returns array of length n_paths", S_T.shape == (N,), f"got {S_T.shape}")
check("all strictly positive", bool((S_T > 0).all()))
 
mean_target = S0 * np.exp(R * T)
mean_se = S_T.std(ddof=1) / np.sqrt(N)
z_mean = (S_T.mean() - mean_target) / mean_se
check("E[S_T] = S0*e^(rT) within 4 SE", abs(z_mean) < 4,
      f"mean {S_T.mean():.4f} vs {mean_target:.4f}, z={z_mean:+.2f}")
 
lr = np.log(S_T / S0)
check("log-return mean = (r-0.5s^2)T", abs(lr.mean() - (R - 0.5 * SIGMA**2) * T) < 0.002,
      f"got {lr.mean():.5f} vs {(R - 0.5*SIGMA**2)*T:.5f}")
check("log-return std = sigma*sqrt(T)", abs(lr.std(ddof=1) - SIGMA * np.sqrt(T)) < 0.002,
      f"got {lr.std(ddof=1):.5f} vs {SIGMA*np.sqrt(T):.5f}")
 
# ----------------------------------------------------------------------
print("\n6. Monte Carlo vs Black-Scholes across strikes")
 
ladder_K = [80.0, 90.0, 100.0, 110.0, 120.0, 150.0]
mc_p, mc_se, bs_p = [], [], []
for k in ladder_K:
    rng = np.random.default_rng(1000 + int(k))
    p, se = mc_call_price(S0, k, R, SIGMA, T, 1_000_000, rng)
    mc_p.append(p)
    mc_se.append(se)
    bs_p.append(bs_call_price(S0, k, R, SIGMA, T))
 
mc_p, mc_se, bs_p = np.array(mc_p), np.array(mc_se), np.array(bs_p)
z = (mc_p - bs_p) / mc_se
 
print(f"    {'K':>6} {'MC':>10} {'SE':>8} {'BS':>10} {'z':>7} {'rel SE':>8}")
for i, k in enumerate(ladder_K):
    print(f"    {k:>6.0f} {mc_p[i]:>10.4f} {mc_se[i]:>8.4f} {bs_p[i]:>10.4f} "
          f"{z[i]:>+7.2f} {mc_se[i]/mc_p[i]*100:>7.2f}%")
 
check("all |z| < 3", bool(np.all(np.abs(z) < 3)), f"max |z| = {np.abs(z).max():.2f}")
 
# ----------------------------------------------------------------------
print("\n7. Standard error scales as 1/sqrt(N)")
 
Ns = np.array([10_000, 100_000, 1_000_000])
ses = []
for n in Ns:
    rng = np.random.default_rng(7)
    ses.append(mc_call_price(S0, K, R, SIGMA, T, int(n), rng)[1])
ses = np.array(ses)
se_slope = np.polyfit(np.log(Ns), np.log(ses), 1)[0]
check("SE log-log slope ~ -0.5", abs(se_slope + 0.5) < 0.05, f"got {se_slope:+.4f}")
 
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"  {len(passed)} passed, {len(failed)} failed")
if failed:
    print("\n  FAILED:")
    for f in failed:
        print(f"    - {f}")
print("=" * 60)
 
# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Steps 0-4 validation", fontsize=13, fontweight="bold")
 
# (a) terminal price distribution
ax = axes[0, 0]
ax.hist(S_T, bins=200, range=(0, 300), density=True, color="steelblue",
        alpha=0.75, edgecolor="none")
x = np.linspace(1, 300, 800)
mu_l, sd_l = (R - 0.5 * SIGMA**2) * T, SIGMA * np.sqrt(T)
pdf = np.exp(-((np.log(x / S0) - mu_l) ** 2) / (2 * sd_l**2)) / (x * sd_l * np.sqrt(2 * np.pi))
ax.plot(x, pdf, color="darkorange", lw=2, label="lognormal pdf")
ax.axvline(S0 * np.exp(R * T), color="crimson", ls="--", lw=1.5,
           label=f"$E[S_T]={S0*np.exp(R*T):.1f}$")
ax.axvline(K, color="grey", ls=":", lw=1.5, label=f"$K={K:.0f}$")
ax.set_title("(a) Terminal prices vs theory")
ax.set_xlabel("$S_T$")
ax.set_ylabel("density")
ax.legend(fontsize=8)
 
# (b) payoff distribution
ax = axes[0, 1]
payoffs = np.maximum(S_T - K, 0)
frac_zero = (payoffs == 0).mean()
ax.hist(payoffs[payoffs > 0], bins=150, range=(0, 150), color="seagreen",
        alpha=0.8, edgecolor="none")
ax.set_yscale("log")
ax.set_title(f"(b) Payoffs  ({frac_zero:.1%} expire worthless)")
ax.set_xlabel("$\\max(S_T-K,\\,0)$")
ax.set_ylabel("count (log)")
 
# (c) MC vs BS across strikes
ax = axes[1, 0]
ax.errorbar(ladder_K, mc_p, yerr=1.96 * mc_se, fmt="o", color="steelblue",
            capsize=4, label="Monte Carlo (95% CI)", zorder=3)
kk = np.linspace(75, 155, 200)
ax.plot(kk, [bs_call_price(S0, k, R, SIGMA, T) for k in kk],
        color="darkorange", lw=2, label="Black-Scholes", zorder=2)
ax.set_title("(c) MC vs closed form")
ax.set_xlabel("strike $K$")
ax.set_ylabel("call price")
ax.legend(fontsize=8)
 
# (d) convergence
ax = axes[1, 1]
Ns_conv = np.array([10**3, 10**4, 10**5, 10**6])
errs = []
for n in Ns_conv:
    reps = [abs(mc_call_price(S0, K, R, SIGMA, T, int(n),
                              np.random.default_rng(s))[0] - C) for s in range(20)]
    errs.append(np.mean(reps))
errs = np.array(errs)
slope = np.polyfit(np.log(Ns_conv), np.log(errs), 1)[0]
ax.loglog(Ns_conv, errs, "o-", color="steelblue", label=f"mean |error|, slope {slope:+.3f}")
ax.loglog(Ns_conv, errs[0] * (Ns_conv / Ns_conv[0]) ** -0.5, "--",
          color="crimson", label="reference $N^{-1/2}$")
ax.set_title("(d) Convergence (20 seeds per N)")
ax.set_xlabel("paths $N$")
ax.set_ylabel("|MC - BS|")
ax.legend(fontsize=8)
 
plt.tight_layout()
plt.savefig("validation.png", dpi=130)
print("\nSaved validation.png")
plt.show()
 

