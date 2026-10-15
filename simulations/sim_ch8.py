"""Chapter 8 simulations (pure standard library, seeded)."""
import random, math, statistics as st
from statlib import (norm_cdf, norm_quantile, t_quantile, t_sf_two_sided,
                     binom_pmf, binom_sf)

random.seed(808)

def hdr(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)

def tstat(x, mu0):
    n = len(x)
    m = sum(x) / n
    s = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    return (m - mu0) / (s / math.sqrt(n))

# ============================== 8.3.1 ==============================
hdr("8.3.1 POWER FUNCTION of the one-sample t-test, H0: mu=0 vs H1: mu!=0, n=10, alpha=0.05")
n, alpha, R = 10, 0.05, 100000
tc = t_quantile(1 - alpha / 2, n - 1)
print(f"  critical value |t| > t_(0.975,9) = {tc:.4f}")
for mu in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
    rej = 0
    for _ in range(R):
        x = [random.gauss(mu, 1.0) for _ in range(n)]
        if abs(tstat(x, 0.0)) > tc:
            rej += 1
    tag = "  <- size: equals alpha exactly" if mu == 0 else ""
    print(f"  mu={mu:4.2f}: power = {rej/R:.4f}{tag}")

# ============================== 8.3.4 ==============================
hdr("8.3.4 p-VALUES are Uniform(0,1) under H0, and concentrate near 0 under H1")
R = 200000
for mu in (0.0, 0.5, 1.0):
    ps = []
    for _ in range(R):
        x = [random.gauss(mu, 1.0) for _ in range(n)]
        ps.append(t_sf_two_sided(tstat(x, 0.0), n - 1))
    ps.sort()
    q = lambda a: ps[int(a * R)]
    print(f"  mu={mu:3.1f}: q10={q(.10):.4f} q25={q(.25):.4f} q50={q(.50):.4f} "
          f"q75={q(.75):.4f} q90={q(.90):.4f}")
    print(f"           P(p<=0.05)={sum(1 for v in ps if v <= 0.05)/R:.4f}  "
          f"P(p<=0.10)={sum(1 for v in ps if v <= 0.10)/R:.4f}  "
          f"P(p<=0.50)={sum(1 for v in ps if v <= 0.50)/R:.4f}")
print("  (under H0 the quantiles sit on the diagonal: q10=0.10, q50=0.50, ... as the theorem requires)")

# ============================== 8.3.2 (a) ==============================
hdr("8.3.2 NEYMAN-PEARSON: the LR test beats every competitor of the same size")
n, R = 10, 200000
mu1 = 0.5
print(f"  Simple vs simple: H0: mu=0 vs H1: mu={mu1}, sigma=1 known, n={n}, alpha=0.05.")
print("  The likelihood ratio is monotone in Xbar, so the NP test rejects for large Xbar.")
c_np = norm_quantile(0.95) / math.sqrt(n)
c_x1 = norm_quantile(0.95)
# critical value for the median test, calibrated by simulation under H0
meds = []
for _ in range(R):
    x = sorted(random.gauss(0.0, 1.0) for _ in range(n))
    meds.append(0.5 * (x[n // 2 - 1] + x[n // 2]))
meds.sort()
c_med = meds[int(0.95 * R)]
print(f"  critical values: Xbar > {c_np:.4f} | median > {c_med:.4f} (calibrated) | X1 > {c_x1:.4f}")
size = {"Xbar (NP)": 0, "median": 0, "X1 only": 0}
powr = {"Xbar (NP)": 0, "median": 0, "X1 only": 0}
for _ in range(R):
    x0 = [random.gauss(0.0, 1.0) for _ in range(n)]
    x1 = [random.gauss(mu1, 1.0) for _ in range(n)]
    for lbl, d in (("size", x0), ("power", x1)):
        tgt = size if lbl == "size" else powr
        srt = sorted(d)
        med = 0.5 * (srt[n // 2 - 1] + srt[n // 2])
        if sum(d) / n > c_np:
            tgt["Xbar (NP)"] += 1
        if med > c_med:
            tgt["median"] += 1
        if d[0] > c_x1:
            tgt["X1 only"] += 1
print("  test        |  achieved size  |  power at mu=0.5   | analytic power")
an = {"Xbar (NP)": 1 - norm_cdf((c_np - mu1) * math.sqrt(n)),
      "median": float('nan'),
      "X1 only": 1 - norm_cdf(c_x1 - mu1)}
for k in ("Xbar (NP)", "median", "X1 only"):
    a = f"{an[k]:.4f}" if not math.isnan(an[k]) else "   --  "
    print(f"  {k:11s} |     {size[k]/R:.4f}      |      {powr[k]/R:.4f}        |    {a}")
print("  -> all three have size 0.05; the NP (Xbar) test has the largest power, as the lemma requires.")

# ============================== 8.3.2 (b) ==============================
hdr("8.3.2 KARLIN-RUBIN: the one-sided UMP test dominates at EVERY alternative")
print(f"  H0: mu<=0 vs H1: mu>0, sigma=1 known, n={n}, alpha=0.05. UMP rejects Xbar > {c_np:.4f}.")
R = 120000
print("   mu   |  UMP (Xbar)  |  median test  |  X1-only test")
for mu in (0.0, 0.2, 0.4, 0.6, 0.8):
    a = b = c = 0
    for _ in range(R):
        x = [random.gauss(mu, 1.0) for _ in range(n)]
        srt = sorted(x)
        med = 0.5 * (srt[n // 2 - 1] + srt[n // 2])
        if sum(x) / n > c_np:
            a += 1
        if med > c_med:
            b += 1
        if x[0] > c_x1:
            c += 1
    print(f"  {mu:4.2f}  |    {a/R:.4f}    |    {b/R:.4f}     |    {c/R:.4f}")
print("  -> the UMP column is largest in every row: uniform dominance, not just at one alternative.")

# ============================== 8.3.2 (c) ==============================
hdr("8.3.2 NO UMP TEST EXISTS for the two-sided problem: the power curves cross")
R = 120000
c_two = norm_quantile(0.975) / math.sqrt(n)
print(f"  one-sided test: Xbar > {c_np:.4f}    two-sided test: |Xbar| > {c_two:.4f}   (both size 0.05)")
print("   mu    |  one-sided power  |  two-sided power")
for mu in (-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0):
    a = b = 0
    for _ in range(R):
        xb = sum(random.gauss(mu, 1.0) for _ in range(n)) / n
        if xb > c_np:
            a += 1
        if abs(xb) > c_two:
            b += 1
    print(f"  {mu:5.2f}  |      {a/R:.4f}       |      {b/R:.4f}")
print("  -> neither dominates: one-sided wins for mu>0, two-sided wins for mu<0. So in the")
print("     two-sided problem no single test is uniformly most powerful.")

# ============================== 8.3.3 ==============================
hdr("8.3.3 IUT: an intersection-union test has size <= alpha, attained only in a limit")
print("  H0: mu1<=0 OR mu2<=0   vs   H1: both > 0.  Reject iff BOTH z-tests reject at 0.05.")
R, n2 = 120000, 10
cz = norm_quantile(0.95) / math.sqrt(n2)
for m1, m2 in ((0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (0.0, 3.0), (-0.5, 3.0)):
    rej = 0
    for _ in range(R):
        a = sum(random.gauss(m1, 1.0) for _ in range(n2)) / n2
        b = sum(random.gauss(m2, 1.0) for _ in range(n2)) / n2
        if a > cz and b > cz:
            rej += 1
    print(f"  (mu1,mu2)=({m1:4.1f},{m2:4.1f}) in H0: rejection probability = {rej/R:.4f}")
print(f"  -> at (0,0) it is about alpha^2={0.05**2:.4f}; it climbs to alpha={0.05:.2f} only as the")
print("     second mean runs off to infinity. The size is alpha but the test is conservative inside H0.")

# ============================== 8.2.1 ==============================
hdr("8.2.1 THE LRT FOR A NORMAL MEAN IS THE t-TEST: lambda = (1 + t^2/(n-1))^(-n/2)")
R = 100000
worst = 0.0
for _ in range(R):
    x = [random.gauss(random.choice([0.0, 0.6]), 1.0) for _ in range(n)]
    m = sum(x) / n
    ss0 = sum(v * v for v in x)              # sup over sigma^2 under H0: mu=0
    ss1 = sum((v - m) ** 2 for v in x)       # sup over (mu,sigma^2)
    lam = (ss1 / ss0) ** (n / 2.0)
    t = tstat(x, 0.0)
    pred = (1.0 + t * t / (n - 1)) ** (-n / 2.0)
    worst = max(worst, abs(lam - pred))
print(f"  max |lambda - (1+t^2/(n-1))^(-n/2)| over {R} random samples = {worst:.3e}")
print("  -> an exact algebraic identity, so 'reject when lambda <= c' IS 'reject when |t| >= c''.")

# ============================== 8.3.1 discreteness ==============================
hdr("8.3.1 DISCRETE TESTS CANNOT HIT A NOMINAL SIZE: Binomial(20,p), H0: p=0.5 vs H1: p>0.5")
n3 = 20
print("   k  |  P(X >= k | p=0.5)")
for k in range(12, 18):
    print(f"  {k:3d} |      {binom_sf(k, n3, 0.5):.5f}")
kstar = min(k for k in range(n3 + 1) if binom_sf(k, n3, 0.5) <= 0.05)
print(f"  smallest k with P(X>=k|0.5) <= 0.05 is k={kstar}, giving achieved size "
      f"{binom_sf(kstar, n3, 0.5):.5f}")
print(f"  the nominal 0.05 is unreachable: k={kstar-1} would give {binom_sf(kstar-1, n3, 0.5):.5f} > 0.05.")
R = 200000
rej = sum(1 for _ in range(R)
          if sum(1 for _ in range(n3) if random.random() < 0.5) >= kstar)
print(f"  simulated size at p=0.5: {rej/R:.5f}   (matches the exact {binom_sf(kstar, n3, 0.5):.5f})")
print(f"  -> real size {binom_sf(kstar, n3, 0.5):.3f}, not 0.05: the test spends only "
      f"{binom_sf(kstar, n3, 0.5)/0.05:.0%} of its error budget, and loses power accordingly.")

# ============================== 8.2.2 ==============================
hdr("8.2.2 BAYESIAN TESTS: a p-value is NOT P(H0 | data)  (Jeffreys-Lindley)")
print("  H0: mu=0 vs H1: mu!=0, sigma=1 known, prior P(H0)=0.5, and mu ~ N(0,tau^2) under H1.")
print("  Observe exactly z = 1.95996, i.e. a two-sided p-value of 0.05, for each n.")
z = norm_quantile(0.975)
print("      n   |  two-sided p  |  Bayes factor B01  |  P(H0 | data)")
for nn in (10, 100, 1000, 10000, 100000):
    r = 1.0 * nn                     # tau^2 / (sigma^2/n) with tau=sigma=1
    B01 = math.sqrt(1 + r) * math.exp(-z * z * r / (2 * (1 + r)))
    print(f"  {nn:7d} |     0.0500    |      {B01:8.4f}      |     {B01/(1+B01):.4f}")
print("  -> at n=10 a p-value of 0.05 leaves P(H0|data) about 0.37, not 0.05; by n=100000 the")
print("     SAME p-value is strong evidence FOR H0 (P(H0|data)=0.98). The p-value and the")
print("     posterior probability of H0 answer different questions and can point opposite ways.")
