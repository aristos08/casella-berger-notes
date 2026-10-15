"""Chapter 10 simulations (pure standard library, seeded)."""
import random, math, statistics as st
from statlib import (norm_cdf, norm_pdf, norm_quantile, chi2_cdf, chi2_quantile,
                     beta_quantile, binom_pmf)

random.seed(1010)

def hdr(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)

def rpois(lam):
    L = math.exp(-lam); k = 0; p = 1.0
    while True:
        p *= random.random()
        if p <= L:
            return k
        k += 1

def median(x):
    s = sorted(x); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

def rlaplace(b=1.0):
    u = random.random() - 0.5
    return -b * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))

# ====================== 10.1.1 / 10.1.2 ======================
hdr("10.1.1-2 MLE of tau=exp(-lambda): biased, but asymptotically efficient")
# MLE is exp(-Xbar); Ch.7 UMVUE is ((n-1)/n)^T.  Exact moments available for both.
lam = 1.0
tau = math.exp(-lam)
print(f"  target tau = exp(-1) = {tau:.6f};  CRLB = lambda*exp(-2*lambda)/n")
print("    n   | E[MLE]   bias     Var(MLE)   MSE(MLE)  | MSE(UMVUE) |  CRLB    | MSE(MLE)/CRLB")
for n in (5, 10, 40, 200, 1000):
    # exact: T ~ Poisson(n*lam); E[exp(-aT)] = exp(n*lam*(exp(-a)-1))
    e1 = math.exp(n * lam * (math.exp(-1.0 / n) - 1))
    e2 = math.exp(n * lam * (math.exp(-2.0 / n) - 1))
    var_mle = e2 - e1 * e1
    mse_mle = var_mle + (e1 - tau) ** 2
    a = (n - 1) / n
    mse_umv = math.exp(lam * (1 - 2 * n) / n) - math.exp(-2 * lam)
    crlb = lam * math.exp(-2 * lam) / n
    print(f"  {n:5d} | {e1:.5f} {e1-tau:+.5f}  {var_mle:.6f}  {mse_mle:.6f} |  {mse_umv:.6f} | {crlb:.6f} |   {mse_mle/crlb:.4f}")
print("  -> the MLE is biased upward and (here) slightly worse than the UMVUE at small n,")
print("     but both ratios to the CRLB tend to 1: asymptotic efficiency, not finite-sample")
print("     optimality, is what the MLE delivers.")
R, n = 200000, 10
sim = []
for _ in range(R):
    t = sum(rpois(lam) for _ in range(n))
    sim.append(math.exp(-t / n))
e1 = math.exp(n * lam * (math.exp(-1.0 / n) - 1))
print(f"  simulation check at n=10 ({R} reps): E[MLE]={st.mean(sim):.5f} (exact {e1:.5f}), "
      f"Var={st.variance(sim):.6f} (exact {math.exp(n*lam*(math.exp(-2.0/n)-1))-e1*e1:.6f})")

hdr("10.1.2 ASYMPTOTIC NORMALITY of the MLE: sqrt(n)(theta_hat - theta) -> N(0, 1/I)")
# exponential rate: MLE = 1/Xbar, I(theta)=1/theta^2, so avar = theta^2
th = 2.0
print(f"  Exp(rate theta={th}): MLE = 1/Xbar, I(theta)=1/theta^2, asymptotic var = theta^2 = {th*th}")
print("  Exact finite-n moments are available: E[1/Xbar]=n*theta/(n-1) and")
print("  Var(1/Xbar)=n^2*theta^2/((n-1)^2*(n-2)), so the drift and spread are predictable.")
print("    n    | sim mean sqrt(n)*bias | exact | sim n*Var | exact n*Var | limit | P(|.|<1.96 se)")
for n in (10, 50, 200, 1000):
    R = 60000
    zs = []
    for _ in range(R):
        s = sum(random.expovariate(th) for _ in range(n))
        zs.append(math.sqrt(n) * (n / s - th))
    ex_bias = math.sqrt(n) * th / (n - 1)
    ex_nvar = n * (n * n * th * th / ((n - 1) ** 2 * (n - 2)))
    cov = sum(1 for v in zs if abs(v / th) < 1.96) / R
    print(f"  {n:5d}  |       {st.mean(zs):+.4f}        | {ex_bias:+.4f} |  {st.variance(zs):.4f}  "
          f"|   {ex_nvar:.4f}    |  {th*th:.1f}  |     {cov:.4f}")
print("  -> Monte Carlo error is larger here than elsewhere on this page because 1/T is")
print("     heavy-tailed at small n; the exact columns are the ones to trust.")

# ====================== 10.1.3 ======================
hdr("10.1.3 ASYMPTOTIC RELATIVE EFFICIENCY: mean vs median depends on the tails")
print("  Sample median avar = 1/(4 n f(m)^2).  Normal: pi*sigma^2/(2n), so ARE(med,mean)=2/pi.")
print(f"  2/pi = {2/math.pi:.4f};  Laplace: median avar = b^2/n vs mean 2b^2/n, ARE = 2.0")
for lbl, draw, are in (("Normal(0,1)   ", lambda: random.gauss(0, 1), 2 / math.pi),
                       ("Laplace(0,b=1)", lambda: rlaplace(1.0), 2.0)):
    print(f"   {lbl}  theory ARE(median vs mean) = {are:.4f}")
    for n in (11, 51, 201):
        R = 40000
        mn, md = [], []
        for _ in range(R):
            x = [draw() for _ in range(n)]
            mn.append(sum(x) / n)
            md.append(median(x))
        print(f"      n={n:4d}: var(mean)={st.variance(mn):.5f}  var(median)={st.variance(md):.5f}  "
              f"ratio var(mean)/var(median)={st.variance(mn)/st.variance(md):.4f}")

# ====================== 10.2.1 ======================
hdr("10.2.1 ROBUSTNESS: 10% contamination from N(0,100) destroys the mean")
n, R = 50, 60000
eps = 0.10
def trimmed(x, prop=0.2):
    s = sorted(x); k = int(prop * len(s))
    core = s[k:len(s) - k]
    return sum(core) / len(core)
res = {"mean": [], "median": [], "20% trimmed": []}
for _ in range(R):
    x = [random.gauss(0, 10) if random.random() < eps else random.gauss(0, 1) for _ in range(n)]
    res["mean"].append(sum(x) / n)
    res["median"].append(median(x))
    res["20% trimmed"].append(trimmed(x))
print(f"  n={n}, mixture 0.9*N(0,1) + 0.1*N(0,100); true centre 0.  MSE over {R} reps:")
for k in ("mean", "median", "20% trimmed"):
    v = res[k]
    print(f"     {k:12s}: MSE={sum(z*z for z in v)/R:.5f}  sd={st.stdev(v):.5f}")
print("  (for comparison, with NO contamination the mean's MSE would be 1/n = 0.0200)")
print("  -> the mean's variance is inflated by the 10% of draws with sd 10; the median and")
print("     trimmed mean barely notice. Breakdown point: mean 0, median 0.5, 20% trimmed 0.2.")

# ====================== 10.2.2 ======================
hdr("10.2.2 M-ESTIMATORS: Huber's psi, and the 95%-efficiency constant k=1.345")
k_hub = 1.345
def psi(u, k=k_hub):
    return max(-k, min(k, u))
def huber_loc(x, k=k_hub, iters=200):
    lo, hi = min(x) - 1.0, max(x) + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if sum(psi(v - mid, k) for v in x) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
# analytic asymptotic variance at the normal: E[psi^2]/(E[psi'])^2
Epsi2 = (2 * norm_cdf(k_hub) - 1) - 2 * k_hub * norm_pdf(k_hub) + 2 * k_hub ** 2 * (1 - norm_cdf(k_hub))
Edpsi = 2 * norm_cdf(k_hub) - 1
avar = Epsi2 / Edpsi ** 2
print(f"  k={k_hub}: E[psi^2]={Epsi2:.5f}  E[psi']={Edpsi:.5f}  avar={avar:.5f}  "
      f"ARE vs mean = {1/avar:.4f}")
print("  -> the constant 1.345 is chosen to buy 95% efficiency at the normal while bounding")
print("     the influence of outliers.")
for n in (20, 100, 400):
    R = 30000
    hu, mn = [], []
    for _ in range(R):
        x = [random.gauss(0, 1) for _ in range(n)]
        hu.append(huber_loc(x))
        mn.append(sum(x) / n)
    print(f"    normal data, n={n:4d}: n*var(Huber)={n*st.variance(hu):.4f} (limit {avar:.4f})  "
          f"n*var(mean)={n*st.variance(mn):.4f} (limit 1)")
R = 40000
hu, mn, md = [], [], []
for _ in range(R):
    x = [random.gauss(0, 10) if random.random() < eps else random.gauss(0, 1) for _ in range(50)]
    hu.append(huber_loc(x)); mn.append(sum(x) / 50); md.append(median(x))
print(f"  contaminated data (n=50, 10% from N(0,100)): MSE Huber={sum(v*v for v in hu)/R:.5f}  "
      f"mean={sum(v*v for v in mn)/R:.5f}  median={sum(v*v for v in md)/R:.5f}")

# ====================== 10.3.1 ======================
hdr("10.3.1 ASYMPTOTIC DISTRIBUTION OF THE LRT: -2 log lambda -> chi^2_1")
def lrt_bern(x_sum, n, p0):
    ph = x_sum / n
    if ph in (0.0, 1.0):
        t1 = 0.0 if ph == 0 else n * math.log(ph / p0)
        if ph == 0:
            return 2 * n * math.log((1 - ph) / (1 - p0)) if (1 - p0) > 0 else 0.0
        return 2 * t1
    return 2 * n * (ph * math.log(ph / p0) + (1 - ph) * math.log((1 - ph) / (1 - p0)))
c95 = chi2_quantile(0.95, 1)
print(f"  Bernoulli, H0: p=p0.  chi^2_(0.95,1) = {c95:.4f}")
print("    p0    |   n   |  P(-2logL > c95)  |  mean(-2logL)  (limit 1.0)")
for p0 in (0.5, 0.1):
    for n in (10, 25, 100, 500):
        R = 100000
        rej = 0; tot = 0.0
        for _ in range(R):
            s = sum(1 for _ in range(n) if random.random() < p0)
            v = lrt_bern(s, n, p0)
            tot += v
            if v > c95:
                rej += 1
        print(f"   {p0:5.2f}  | {n:5d} |      {rej/R:.4f}       |     {tot/R:.4f}")
print("  -> the chi^2_1 approximation is good by n=100 at p0=0.5 but still visibly off at")
print("     p0=0.1 with n=25: the relevant quantity is n*p0, not n.")

# ====================== 10.3.2 ======================
hdr("10.3.2 WALD vs SCORE vs LRT: three asymptotically equivalent tests, three finite-n sizes")
print("  Bernoulli, H0: p=p0 vs two-sided, nominal 5%.  EXACT sizes by summation over x.")
z = norm_quantile(0.975)
print("  Note: at x=0 or x=n the Wald statistic is 0/0. Treating it as a rejection (statistic")
print("  = infinity) is the usual convention; the column 'Wald*' instead treats those outcomes")
print("  as no-decision, to separate the degeneracy from genuine size distortion.")
print("    p0    |   n   |   Wald   |  Wald*   |  Score   |   LRT    | P(x=0 or n)")
for p0 in (0.5, 0.1, 0.03):
    for n in (20, 50, 200):
        sz_w = sz_w2 = sz_s = sz_l = 0.0
        p_deg = binom_pmf(0, n, p0) + binom_pmf(n, n, p0)
        for x in range(n + 1):
            pr = binom_pmf(x, n, p0)
            ph = x / n
            se_w = math.sqrt(ph * (1 - ph) / n)
            if se_w > 0:
                if abs(ph - p0) / se_w > z:
                    sz_w += pr
                    sz_w2 += pr
            else:
                sz_w += pr            # convention: degenerate case rejects
            if abs(ph - p0) / math.sqrt(p0 * (1 - p0) / n) > z:
                sz_s += pr
            if lrt_bern(x, n, p0) > c95:
                sz_l += pr
        print(f"   {p0:5.2f}  | {n:5d} |  {sz_w:.4f}  |  {sz_w2:.4f}  |  {sz_s:.4f}  |  "
              f"{sz_l:.4f}  |   {p_deg:.4f}")
print("  -> all three share the chi^2_1 limit, yet at finite n they differ sharply. Reading the")
print("     nine rows: Score stays in [0.021, 0.065]; LRT ranges over [0.017, 0.133]; Wald with")
print("     the usual convention reaches 0.544, almost entirely from the degenerate x=0 outcome")
print("     (compare the last column). Excluding that, Wald* is not uniformly liberal but")
print("     ERRATIC, from 0.0003 to 0.116. Score is the only one that is reliably near nominal.")
print("     Sizes are non-monotone in n because the statistic is discrete (lattice effects).")

# ====================== 10.1.4 ======================
hdr("10.1.4 BOOTSTRAP STANDARD ERRORS: works for smooth functionals")
n, B, OUT = 25, 400, 400
print(f"  n={n}, B={B} resamples, {OUT} outer replications, data N(0,1).")
for lbl, stat, truth in (("mean  ", lambda v: sum(v) / len(v), 1.0 / math.sqrt(n)),
                         ("median", median, None)):
    boots, plug = [], []
    for _ in range(OUT):
        x = [random.gauss(0, 1) for _ in range(n)]
        bs = []
        for _ in range(B):
            r = [x[random.randrange(n)] for _ in range(n)]
            bs.append(stat(r))
        boots.append(st.stdev(bs))
        plug.append(stat(x))
    emp = st.stdev(plug)
    th = f"{truth:.4f}" if truth else "   --  "
    print(f"   {lbl}: mean bootstrap SE = {st.mean(boots):.4f}   actual sd of the statistic = {emp:.4f}   theory {th}")
print("  -> the bootstrap SE tracks the true sampling sd without any formula, including for")
print("     the median where the delta method would need the density at the median.")

hdr("10.1.4 ... AND FAILS for the maximum: the bootstrap cannot resample an extreme")
n = 50
R = 200000
hits = 0
for _ in range(R):
    # probability that a single bootstrap resample reproduces the sample maximum
    if any(random.randrange(n) == 0 for _ in range(n)):
        hits += 1
print(f"  n={n}: P(a bootstrap resample contains the sample max) = {hits/R:.4f}")
print(f"        exact 1-(1-1/n)^n = {1-(1-1/n)**n:.4f}   limit 1-1/e = {1-math.exp(-1):.4f}")
print("  So the bootstrap distribution of the max has an atom of mass about 0.632 at the")
print("  observed max. For Uniform(0,theta) the true limit law of n(theta - X_(n))/theta is")
print("  Exponential(1), which is continuous -- so the bootstrap limit is simply the wrong law.")
R, n = 100000, 50
true_pivot = []
for _ in range(R):
    mx = max(random.random() for _ in range(n))
    true_pivot.append(n * (1.0 - mx))
true_pivot.sort()
x = [random.random() for _ in range(n)]
mx = max(x)
boot_pivot = []
for _ in range(R):
    bm = max(x[random.randrange(n)] for _ in range(n))
    boot_pivot.append(n * (mx - bm) / 1.0)
atom = sum(1 for v in boot_pivot if v == 0.0) / R
print(f"  true pivot n(theta-X_(n)): P(=0)=0.0000, median={true_pivot[R//2]:.4f} "
      f"(Exp(1) median {math.log(2):.4f})")
print(f"  bootstrap pivot n(X_(n)-X*_(n)): P(=0)={atom:.4f}  <- a point mass that should not be there")

# ====================== 10.4 ======================
hdr("10.4 LARGE-SAMPLE INTERVALS: Wilson IS the inverted score test (and that is why it wins)")
nb, alpha = 20, 0.05
def wald_i(x, nn=nb):
    ph = x / nn; h = z * math.sqrt(ph * (1 - ph) / nn); return ph - h, ph + h
def wilson_i(x, nn=nb):
    ph = x / nn
    c = (ph + z * z / (2 * nn)) / (1 + z * z / nn)
    h = (z / (1 + z * z / nn)) * math.sqrt(ph * (1 - ph) / nn + z * z / (4 * nn * nn))
    return c - h, c + h
def score_inverted(x, nn=nb):
    """solve |phat-p0| / sqrt(p0(1-p0)/n) = z for p0 by the quadratic formula"""
    ph = x / nn
    A = nn + z * z
    Bq = -(2 * nn * ph + z * z)
    C = nn * ph * ph
    disc = Bq * Bq - 4 * A * C
    r = math.sqrt(max(disc, 0.0))
    return (-Bq - r) / (2 * A), (-Bq + r) / (2 * A)
worst = max(max(abs(wilson_i(x)[0] - score_inverted(x)[0]),
                abs(wilson_i(x)[1] - score_inverted(x)[1])) for x in range(nb + 1))
print(f"  max |Wilson endpoint - inverted-score endpoint| over x=0..{nb}: {worst:.3e}")
def lrt_i(x, nn=nb):
    lo, hi = 1e-12, 1 - 1e-12
    ph = x / nn
    a, b = lo, max(ph, lo)
    for _ in range(200):
        m = 0.5 * (a + b)
        if lrt_bern(x, nn, m) > c95:
            a = m
        else:
            b = m
    L = 0.5 * (a + b)
    a, b = min(ph, hi), hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if lrt_bern(x, nn, m) > c95:
            b = m
        else:
            a = m
    U = 0.5 * (a + b)
    return L, U
methods = {"Wald": [wald_i(x) for x in range(nb + 1)],
           "Score (Wilson)": [wilson_i(x) for x in range(nb + 1)],
           "LRT-inverted": [lrt_i(x) for x in range(nb + 1)],
           "Clopper-Pearson": [(0.0 if x == 0 else beta_quantile(alpha/2, x, nb-x+1),
                                1.0 if x == nb else beta_quantile(1-alpha/2, x+1, nb-x))
                               for x in range(nb + 1)]}
def cov(iv, p):
    return sum(binom_pmf(x, nb, p) for x in range(nb + 1) if iv[x][0] <= p <= iv[x][1])
def mlen(iv, p):
    return sum(binom_pmf(x, nb, p) * (iv[x][1] - iv[x][0]) for x in range(nb + 1))
grid = [0.001 + 0.001 * i for i in range(999)]
print(f"\n  n={nb}, nominal 95%, EXACT coverage over p in (0.001,0.999):")
print("     method            |  min cov  |  mean cov  |  mean length")
for m in ("Wald", "Score (Wilson)", "LRT-inverted", "Clopper-Pearson"):
    cs = [cov(methods[m], p) for p in grid]
    ls = [mlen(methods[m], p) for p in grid]
    print(f"     {m:18s}|  {min(cs):.4f}   |   {st.mean(cs):.4f}   |    {st.mean(ls):.4f}")
print("  -> the Wald interval uses the variance estimated AT the estimate; the score interval")
print("     uses the variance under each candidate p0, which is correct under that hypothesis.")
print("     That single difference explains Chapter 9's finding.")
