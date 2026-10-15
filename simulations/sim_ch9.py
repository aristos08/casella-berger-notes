"""Chapter 9 simulations (pure standard library, seeded).

Where coverage can be computed exactly (discrete models, and the Uniform
conditional-coverage example) it is computed exactly rather than simulated.
"""
import random, math, statistics as st
from statlib import (norm_cdf, norm_pdf, norm_quantile, t_quantile, t_sf_two_sided,
                     chi2_quantile, beta_quantile, binom_pmf,
                     gamma_cdf, gamma_pdf, gamma_quantile)

random.seed(909)

def hdr(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)

def meansd(x):
    n = len(x)
    m = sum(x) / n
    s = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    return m, s

# ============================== 9.1 / 9.3.1 ==============================
hdr("9.1 THE INTERVAL IS RANDOM, THETA IS FIXED: t-interval, N(0,1), n=10, 95%")
n, R = 10, 200000
tc = t_quantile(0.975, n - 1)
cov = 0
lens = []
first8 = []
misses = []
miss200 = 0
for i in range(R):
    x = [random.gauss(0.0, 1.0) for _ in range(n)]
    m, s = meansd(x)
    h = tc * s / math.sqrt(n)
    covers = abs(m) <= h
    if covers:
        cov += 1
    lens.append(2 * h)
    if i < 8:
        first8.append((m - h, m + h, covers))
    if i < 200 and not covers:
        miss200 += 1
        if len(misses) < 3:
            misses.append((i, m - h, m + h))
print(f"  critical value t_(0.975,9) = {tc:.4f}")
print(f"  coverage over {R} samples = {cov/R:.4f}   (exact theory 0.9500)")
print(f"  mean length = {st.mean(lens):.4f}, sd of length = {st.stdev(lens):.4f}")
print("  the first 8 realized intervals (true mu = 0):")
for lo, hi, c in first8:
    print(f"     [{lo:+.3f}, {hi:+.3f}]   {'covers' if c else 'MISSES'}")
print(f"  of the first 200 intervals, {miss200} missed. The first few misses:")
for i, lo, hi in misses:
    print(f"     sample #{i+1}: [{lo:+.3f}, {hi:+.3f}]   MISSES (0 is outside)")
print("  -> each interval either contains 0 or does not; 0.95 describes the PROCEDURE,")
print("     not any one of these intervals.")

# ============================== 9.2.1 ==============================
hdr("9.2.1 DUALITY: inverting the t-test reproduces the t-interval exactly")
R = 50000
worst = 0.0
for _ in range(R):
    x = [random.gauss(0.3, 1.5) for _ in range(n)]
    m, s = meansd(x)
    h = tc * s / math.sqrt(n)
    # endpoints of {mu0 : |t(mu0)| <= tc} solved directly
    lo_inv = m - tc * s / math.sqrt(n)
    hi_inv = m + tc * s / math.sqrt(n)
    worst = max(worst, abs(lo_inv - (m - h)), abs(hi_inv - (m + h)))
print(f"  max endpoint discrepancy over {R} samples = {worst:.3e}")
# and confirm the acceptance-region characterisation pointwise on a grid
x = [random.gauss(0.3, 1.5) for _ in range(n)]
m, s = meansd(x)
h = tc * s / math.sqrt(n)
bad = 0
for j in range(4001):
    mu0 = m - 3 * h + 6 * h * j / 4000
    tstat = (m - mu0) / (s / math.sqrt(n))
    in_set = abs(tstat) <= tc
    in_int = (m - h) <= mu0 <= (m + h)
    if in_set != in_int:
        bad += 1
print(f"  grid check on one sample: {bad} of 4001 grid points disagree between")
print("     'the test accepts mu0' and 'mu0 lies in the interval'")

# ============================== 9.2.2 ==============================
hdr("9.2.2 PIVOTS: the chi-square pivot for sigma^2, and why equal tails is not shortest")
sig2 = 4.0
a_lo, a_hi = chi2_quantile(0.025, n - 1), chi2_quantile(0.975, n - 1)
print(f"  equal-tailed: chi2_(0.025,9)={a_lo:.4f}, chi2_(0.975,9)={a_hi:.4f}")
R = 200000
cov_e = 0
len_e = []
for _ in range(R):
    x = [random.gauss(0.0, 2.0) for _ in range(n)]
    ss = sum((v - sum(x) / n) ** 2 for v in x)
    lo, hi = ss / a_hi, ss / a_lo
    if lo <= sig2 <= hi:
        cov_e += 1
    len_e.append(hi - lo)
print(f"  equal-tailed interval [SS/chi2_hi, SS/chi2_lo]: coverage={cov_e/R:.4f}  "
      f"mean length={st.mean(len_e):.4f}")
# shortest 95% interval of the form [SS/b, SS/a]: minimise (1/a - 1/b) s.t. F(b)-F(a)=0.95
from statlib import chi2_cdf
best = None
for i in range(1, 950):
    pl = i / 20000.0                      # lower tail mass
    a = chi2_quantile(pl, n - 1)
    b = chi2_quantile(pl + 0.95, n - 1)
    width = 1.0 / a - 1.0 / b
    if best is None or width < best[0]:
        best = (width, pl, a, b)
w, pl, a, b = best
print(f"  shortest such interval: lower tail mass={pl:.4f} (not 0.025), a={a:.4f}, b={b:.4f}")
cov_s = 0
len_s = []
for _ in range(R):
    x = [random.gauss(0.0, 2.0) for _ in range(n)]
    ss = sum((v - sum(x) / n) ** 2 for v in x)
    lo, hi = ss / b, ss / a
    if lo <= sig2 <= hi:
        cov_s += 1
    len_s.append(hi - lo)
print(f"  shortest interval: coverage={cov_s/R:.4f}  mean length={st.mean(len_s):.4f}  "
      f"({100*(1-st.mean(len_s)/st.mean(len_e)):.1f}% shorter)")

# ============================== 9.2.3 ==============================
hdr("9.2.3 PIVOTING THE CDF, continuous case: exponential scale, exact coverage")
# T = sum X_i ~ Gamma(n, scale theta); F(t|theta) is decreasing in theta.
lam_true = 3.0            # scale
R = 200000
cov = 0
for _ in range(R):
    t = sum(random.expovariate(1.0 / lam_true) for _ in range(n))
    # solve F(t|theta)=0.975 and 0.025 using Gamma(n, rate=1/theta):
    # gamma_cdf(t, n, 1/theta) = p  ->  theta = t / chi-type quantile
    q_hi = gamma_quantile(0.975, n, 1.0)   # quantile of Gamma(n,rate 1)
    q_lo = gamma_quantile(0.025, n, 1.0)
    lo, hi = t / q_hi, t / q_lo
    if lo <= lam_true <= hi:
        cov += 1
print(f"  interval [T/q_(0.975), T/q_(0.025)] with Gamma({n},1) quantiles "
      f"{gamma_quantile(0.025,n,1.0):.4f}, {gamma_quantile(0.975,n,1.0):.4f}")
print(f"  coverage = {cov/R:.4f}   (exact theory 0.9500, since T/theta is a pivot)")

hdr("9.2.3 PIVOTING THE CDF, discrete case: Clopper-Pearson is exact-but-conservative")
nb, alpha = 20, 0.05
def cp_interval(x, nn=nb, al=alpha):
    lo = 0.0 if x == 0 else beta_quantile(al / 2, x, nn - x + 1)
    hi = 1.0 if x == nn else beta_quantile(1 - al / 2, x + 1, nn - x)
    return lo, hi
cp = [cp_interval(x) for x in range(nb + 1)]
print(f"  n={nb}: Clopper-Pearson intervals for a few x")
for x in (0, 1, 5, 10, 19, 20):
    print(f"     x={x:2d}: [{cp[x][0]:.4f}, {cp[x][1]:.4f}]")

# ============================== 9.3.1 ==============================
hdr("9.3.1 EXACT COVERAGE of three binomial intervals (no simulation: exact sums)")
z = norm_quantile(0.975)
def wald(x, nn=nb):
    ph = x / nn
    h = z * math.sqrt(ph * (1 - ph) / nn)
    return ph - h, ph + h
def wilson(x, nn=nb):
    ph = x / nn
    c = (ph + z * z / (2 * nn)) / (1 + z * z / nn)
    h = (z / (1 + z * z / nn)) * math.sqrt(ph * (1 - ph) / nn + z * z / (4 * nn * nn))
    return c - h, c + h
methods = {"Wald": [wald(x) for x in range(nb + 1)],
           "Wilson": [wilson(x) for x in range(nb + 1)],
           "Clopper-Pearson": cp}
def exact_coverage(iv, p, nn=nb):
    return sum(binom_pmf(x, nn, p) for x in range(nn + 1) if iv[x][0] <= p <= iv[x][1])
def exact_meanlen(iv, p, nn=nb):
    return sum(binom_pmf(x, nn, p) * (iv[x][1] - iv[x][0]) for x in range(nn + 1))
print(f"  n={nb}, nominal 95%.  Exact coverage at selected p:")
print("     p    |   Wald   |  Wilson  |  Clopper-Pearson")
for p in (0.02, 0.05, 0.10, 0.20, 0.35, 0.50):
    row = "  ".join(f"{exact_coverage(methods[m], p):.4f}" for m in ("Wald", "Wilson", "Clopper-Pearson"))
    print(f"   {p:5.2f}  |  {row}")
grid = [0.001 + 0.001 * i for i in range(999)]
print("\n  minimum / mean exact coverage over p in (0.001, 0.999):")
for m in ("Wald", "Wilson", "Clopper-Pearson"):
    cs = [exact_coverage(methods[m], p) for p in grid]
    ls = [exact_meanlen(methods[m], p) for p in grid]
    print(f"     {m:16s} min={min(cs):.4f}  mean={st.mean(cs):.4f}  mean length={st.mean(ls):.4f}")
print("  -> Wald can drop to essentially zero coverage; Clopper-Pearson never dips below 0.95")
print("     but pays for it in length; Wilson is the practical compromise.")

hdr("9.3.1 CONDITIONAL COVERAGE: a 95% interval that is sometimes 100%, sometimes under 60%")
# Uniform(theta, theta+1), n=5. R = X_(n)-X_(1), and given R=r the midpoint error is
# Uniform(-(1-r)/2, (1-r)/2) -- proved in Chapter 6. So conditional coverage of
# thetahat +/- c is min(1, 2c/(1-r)).  1-R ~ Beta(2, n-1).
nu = 5
def uncond_cov(c, m=200000):
    # exact by numerical integration against the Beta(2,n-1) density of U = 1-R
    tot = 0.0
    K = 20000
    for i in range(K):
        u = (i + 0.5) / K
        dens = u ** 1 * (1 - u) ** (nu - 2) / (math.gamma(2) * math.gamma(nu - 1) / math.gamma(nu + 1))
        tot += min(1.0, 2 * c / u) * dens / K
    return tot
lo_c, hi_c = 0.0, 0.5
for _ in range(60):
    mid = 0.5 * (lo_c + hi_c)
    if uncond_cov(mid) < 0.95:
        lo_c = mid
    else:
        hi_c = mid
cstar = 0.5 * (lo_c + hi_c)
print(f"  n={nu}: the half-width giving 95% UNCONDITIONAL coverage is c = {cstar:.5f}")
print(f"  check: computed unconditional coverage at that c = {uncond_cov(cstar):.4f}")
thresh = 1.0 - 2 * cstar
print(f"  conditional coverage is exactly 1 as soon as r >= 1-2c = {thresh:.4f}")
R = 300000
bins = [(0.0, 0.30), (0.30, thresh), (thresh, 0.78), (0.78, 1.0)]
tally = {b: [0, 0] for b in bins}
for _ in range(R):
    x = [random.random() for _ in range(nu)]          # theta = 0
    lo, hi = min(x), max(x)
    r = hi - lo
    est = 0.5 * (lo + hi) - 0.5
    for b in bins:
        if b[0] <= r < b[1]:
            tally[b][0] += 1
            if abs(est) <= cstar:
                tally[b][1] += 1
            break
def bin_theory(b):
    """exact conditional coverage averaged over the bin, weighted by the density of R"""
    K = 4000
    num = den = 0.0
    for i in range(K):
        r = b[0] + (b[1] - b[0]) * (i + 0.5) / K
        dens = r ** (nu - 2) * (1 - r)
        num += min(1.0, 2 * cstar / (1 - r)) * dens
        den += dens
    return num / den
print("   range R bin     |  share of samples |  cond. coverage |  theory (bin average)")
for b in bins:
    tot, hit = tally[b]
    print(f"   [{b[0]:.4f},{b[1]:.4f}) |      {tot/R:.4f}       |     {hit/tot:.4f}      |      {bin_theory(b):.4f}")
print("  -> the procedure has exactly 95% coverage overall, yet conditional on a wide observed")
print("     range it covers with probability 1, and on a narrow range far less. Averaging over")
print("     samples you did not get is doing real work here (cf. ancillarity, Chapter 6).")

# ============================== 9.2.4 / 9.3.3 ==============================
hdr("9.2.4 BAYESIAN INTERVALS: credible != confidence, and HPD is shorter than equal-tailed")
# (a) frequentist coverage of the 95% equal-tailed Jeffreys-prior credible interval, binomial
def jeff_interval(x, nn=nb, al=alpha):
    return (beta_quantile(al / 2, x + 0.5, nn - x + 0.5),
            beta_quantile(1 - al / 2, x + 0.5, nn - x + 0.5))
jf = [jeff_interval(x) for x in range(nb + 1)]
print(f"  Jeffreys Beta(1/2,1/2) prior, n={nb}: exact FREQUENTIST coverage of the 95% credible set")
print("     p    |  coverage")
for p in (0.02, 0.05, 0.10, 0.25, 0.50):
    print(f"   {p:5.2f}  |   {exact_coverage(jf, p):.4f}")
cs = [exact_coverage(jf, p) for p in grid]
print(f"  over the grid: min={min(cs):.4f}  mean={st.mean(cs):.4f}")
print("  -> a credible interval has no guarantee of frequentist coverage; here it is close on")
print("     average but dips below 0.95 for some p.")
# (b) HPD vs equal-tailed on the skewed Gamma(5, rate 6) posterior from Chapter 7
k, rate = 5, 6.0
et = (gamma_quantile(0.025, k, rate), gamma_quantile(0.975, k, rate))
# HPD: find level such that the two roots of pdf = level enclose 95% mass
def hpd(level):
    mode = (k - 1) / rate
    lo_a, hi_a = 1e-9, mode
    for _ in range(200):
        mid = 0.5 * (lo_a + hi_a)
        if gamma_pdf(mid, k, rate) < level:
            lo_a = mid
        else:
            hi_a = mid
    a = 0.5 * (lo_a + hi_a)
    lo_b, hi_b = mode, 50.0
    for _ in range(200):
        mid = 0.5 * (lo_b + hi_b)
        if gamma_pdf(mid, k, rate) > level:
            lo_b = mid
        else:
            hi_b = mid
    b = 0.5 * (lo_b + hi_b)
    return a, b, gamma_cdf(b, k, rate) - gamma_cdf(a, k, rate)
lo_l, hi_l = 1e-12, gamma_pdf((k - 1) / rate, k, rate)
for _ in range(200):
    mid = 0.5 * (lo_l + hi_l)
    if hpd(mid)[2] > 0.95:
        lo_l = mid
    else:
        hi_l = mid
a, b, mass = hpd(0.5 * (lo_l + hi_l))
print(f"\n  posterior Gamma(shape=5, rate=6) (the Chapter 7 example):")
print(f"     equal-tailed 95%: [{et[0]:.4f}, {et[1]:.4f}]   length {et[1]-et[0]:.4f}")
print(f"     HPD          95%: [{a:.4f}, {b:.4f}]   length {b-a:.4f}   (mass {mass:.4f})")
print(f"     HPD is {100*(1-(b-a)/(et[1]-et[0])):.2f}% shorter; endpoint densities "
      f"{gamma_pdf(a,k,rate):.5f} vs {gamma_pdf(b,k,rate):.5f} (equal, as HPD requires)")

# ============================== 9.3.2 ==============================
hdr("9.3.2 TEST-RELATED OPTIMALITY: inverting the UMP test minimises FALSE coverage")
# one-sided 95% lower bound for a normal mean, sigma=1 known, n=10.
# UMP-inverted: C(x) = [xbar - z_.95/sqrt(n), inf).  Competitor: based on the median.
R = 200000
zc = norm_quantile(0.95) / math.sqrt(n)
meds = []
for _ in range(R):
    x = sorted(random.gauss(0.0, 1.0) for _ in range(n))
    meds.append(0.5 * (x[n // 2 - 1] + x[n // 2]))
meds.sort()
mc = meds[int(0.95 * R)]
print(f"  UMP-inverted lower bound: Xbar - {zc:.4f}.   Median-based: median - {mc:.4f} (calibrated).")
print("  Both have 95% coverage of the true mu=0. False coverage P(mu' in C) for mu' < 0:")
print("    mu'   |  UMP-inverted  |  median-based")
for mup in (-0.2, -0.4, -0.6, -0.8):
    a1 = a2 = 0
    for _ in range(R // 2):
        x = [random.gauss(0.0, 1.0) for _ in range(n)]
        srt = sorted(x)
        med = 0.5 * (srt[n // 2 - 1] + srt[n // 2])
        if sum(x) / n - zc <= mup:
            a1 += 1
        if med - mc <= mup:
            a2 += 1
    print(f"   {mup:5.2f}  |     {a1/(R//2):.4f}     |    {a2/(R//2):.4f}")
print("  -> the UMP-inverted interval includes false values LESS often at every mu' < 0:")
print("     that is uniformly most accurate, the interval analogue of uniformly most powerful.")

# ============================== 9.3.4 ==============================
hdr("9.3.4 LOSS FUNCTIONS: trading length against coverage")
print("  Interval xbar +/- c/sqrt(n) for a normal mean, sigma=1 known, n=10.")
print("  Loss = length + k*(1 - coverage) = 2c/sqrt(n) + k*(2 - 2*Phi(c)).")
print("  Stationarity: d/dc = 2/sqrt(n) - 2k*phi(c) = 0, i.e. phi(c) = 1/(k*sqrt(n)).")
print("     k    |  optimal c  |  implied coverage  |  loss at optimum")
for kk in (2.0, 5.0, 5.4120, 10.0, 50.0):
    target = 1.0 / (kk * math.sqrt(n))
    if target >= norm_pdf(0.0):
        print(f"   {kk:6.2f} |   (no interior optimum: k too small)")
        continue
    lo_cc, hi_cc = 0.0, 40.0
    for _ in range(300):
        mid = 0.5 * (lo_cc + hi_cc)
        if norm_pdf(mid) > target:
            lo_cc = mid
        else:
            hi_cc = mid
    c = 0.5 * (lo_cc + hi_cc)
    covg = 2 * norm_cdf(c) - 1
    loss = 2 * c / math.sqrt(n) + kk * (2 - 2 * norm_cdf(c))
    print(f"   {kk:6.2f} |   {c:.4f}    |      {covg:.4f}        |     {loss:.4f}")
print(f"  -> the conventional c=1.96 (95%) is the optimum for k={1/(norm_pdf(1.95996)*math.sqrt(n)):.4f}:")
print("     choosing '95%' is implicitly pricing a unit of non-coverage at that many units of length.")
