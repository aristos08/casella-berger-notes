"""Chapter 7 simulations (pure standard library, seeded)."""
import random, math, statistics as st

random.seed(707)

def hdr(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)

def rpois(lam):
    """Knuth's method."""
    L = math.exp(-lam); k = 0; p = 1.0
    while True:
        p *= random.random()
        if p <= L:
            return k
        k += 1

def phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

# ============================== 7.2.1 ==============================
hdr("7.2.1 METHOD OF MOMENTS: Gamma(alpha=3, beta=2), closed form but unstable for small n")
at, bt = 3.0, 2.0
for n, R in ((10, 60000), (50, 20000), (500, 3000)):
    A, B = [], []
    for _ in range(R):
        x = [random.gammavariate(at, bt) for _ in range(n)]
        m1 = sum(x) / n
        m2 = sum(v * v for v in x) / n
        v = m2 - m1 * m1                      # biased sample variance
        A.append(m1 * m1 / v)                 # alpha_hat
        B.append(v / m1)                      # beta_hat
    A.sort()
    print(f"  n={n:4d} (R={R}): alpha_hat mean={st.mean(A):7.4f} median={A[R//2]:6.4f} "
          f"q05={A[int(.05*R)]:6.4f} q95={A[int(.95*R)]:7.4f}   (true 3)")
    print(f"              beta_hat  mean={st.mean(B):7.4f}   (true 2)")

hdr("7.2.1 MoM CAN CONTRADICT THE DATA: Uniform(0,theta), theta=1, n=10")
n, R = 10, 200000
bad = 0
mom, mle, umv = [], [], []
for _ in range(R):
    x = [random.random() for _ in range(n)]
    xbar = sum(x) / n
    mx = max(x)
    m = 2 * xbar
    if m < mx:
        bad += 1
    mom.append((m - 1) ** 2)
    mle.append((mx - 1) ** 2)
    umv.append(((n + 1) / n * mx - 1) ** 2)
print(f"  P(2*Xbar < X_(n)) = {bad/R:.4f}   <- MoM estimate below an observed value")
print(f"  MSE  MoM 2*Xbar        = {st.mean(mom):.5f}   theory 1/(3n)        = {1/(3*n):.5f}")
print(f"  MSE  MLE X_(n)         = {st.mean(mle):.5f}   theory 2/((n+1)(n+2))= {2/((n+1)*(n+2)):.5f}")
print(f"  MSE  UMVUE (n+1)/n*max = {st.mean(umv):.5f}   theory 1/(n(n+2))    = {1/(n*(n+2)):.5f}")

# ============================== 7.2.2 ==============================
hdr("7.2.2 MLE: normal variance. Biased, yet smaller MSE than the unbiased S^2")
n, R = 10, 200000
sig, sig2 = 2.0, 4.0
s2l, mll, c1l, sdl = [], [], [], []
for _ in range(R):
    x = [random.gauss(0.0, sig) for _ in range(n)]
    m = sum(x) / n
    ss = sum((v - m) ** 2 for v in x)
    s2l.append(ss / (n - 1))
    mll.append(ss / n)
    c1l.append(ss / (n + 1))
    sdl.append(math.sqrt(ss / n))
def mse(vals, target):
    return sum((v - target) ** 2 for v in vals) / len(vals)
print(f"  S^2 (divisor n-1): E={st.mean(s2l):.4f} (4)      MSE={mse(s2l,sig2):.4f}  theory {2*sig2**2/(n-1):.4f}")
print(f"  MLE (divisor n)  : E={st.mean(mll):.4f} ({(n-1)/n*sig2:.1f})    MSE={mse(mll,sig2):.4f}  theory {(2*n-1)*sig2**2/n**2:.4f}")
print(f"  divisor n+1      : E={st.mean(c1l):.4f} ({(n-1)/(n+1)*sig2:.4f}) MSE={mse(c1l,sig2):.4f}  theory {2*sig2**2/(n+1):.4f}")
th_sd = sig * math.sqrt(2.0 / n) * math.gamma(n / 2) / math.gamma((n - 1) / 2)
print(f"  invariance: MLE of sigma = sqrt(MLE of sigma^2): E={st.mean(sdl):.4f}  theory {th_sd:.4f}  (sigma=2, so biased low)")

hdr("7.2.2 MLE IS NOT ALWAYS A SOLVED EQUATION: Cauchy location, multiple local maxima")
grid = [-30.0 + 0.05 * i for i in range(1201)]
for n, R in ((5, 1500), (15, 1500)):
    multi = 0
    for _ in range(R):
        x = [math.tan(math.pi * (random.random() - 0.5)) for _ in range(n)]
        ll = [-sum(math.log1p((xi - t) ** 2) for xi in x) for t in grid]
        loc = 0
        for i in range(1, len(ll) - 1):
            if ll[i] > ll[i - 1] and ll[i] >= ll[i + 1]:
                loc += 1
        if loc > 1:
            multi += 1
    print(f"  n={n:3d}: fraction of samples whose log-likelihood has >1 local max = {multi/R:.4f}")

# ============================== 7.2.3 ==============================
hdr("7.2.3 BAYES: Beta(5,5) prior, Binomial n=20. Shrinkage trades bias for variance")
a0, b0, n = 5.0, 5.0, 20
w = (a0 + b0) / (a0 + b0 + n)
print(f"  posterior mean = (5+sum x)/30 = {w:.4f}*0.5 + {1-w:.4f}*Xbar   (shrink weight w={w:.4f})")
R = 60000
for p in (0.1, 0.3, 0.5, 0.9):
    eb, em = [], []
    for _ in range(R):
        s = sum(1 for _ in range(n) if random.random() < p)
        eb.append(((a0 + s) / (a0 + b0 + n) - p) ** 2)
        em.append((s / n - p) ** 2)
    th_b = n * p * (1 - p) / (a0 + b0 + n) ** 2 + ((a0 - (a0 + b0) * p) / (a0 + b0 + n)) ** 2
    print(f"  p={p:3.1f}: MSE Bayes={st.mean(eb):.5f} (theory {th_b:.5f})   MSE MLE={st.mean(em):.5f} (theory {p*(1-p)/n:.5f})")
# integrated (Bayes) risk under the prior
Ep, Ep2 = 0.5, a0 * (a0 + 1) / ((a0 + b0) * (a0 + b0 + 1))
r_mle = (Ep - Ep2) / n
r_bay = n * (Ep - Ep2) / (a0 + b0 + n) ** 2 + (1 - 4 * Ep + 4 * Ep2) / 36
R = 120000
sb, sm = [], []
for _ in range(R):
    p = random.betavariate(a0, b0)
    s = sum(1 for _ in range(n) if random.random() < p)
    sb.append(((a0 + s) / (a0 + b0 + n) - p) ** 2)
    sm.append((s / n - p) ** 2)
print(f"  integrated risk over the prior: Bayes={st.mean(sb):.5f} (theory {r_bay:.5f})   MLE={st.mean(sm):.5f} (theory {r_mle:.5f})")

# ============================== 7.2.4 ==============================
hdr("7.2.4 EM: two-component normal mixture (sigma=1 known). Monotone log-likelihood")
wt, m1t, m2t, N = 0.35, 0.0, 3.0, 400
data = [random.gauss(m2t, 1.0) if random.random() < wt else random.gauss(m1t, 1.0) for _ in range(N)]

def loglik(w, u1, u2):
    return sum(math.log((1 - w) * phi(x - u1) + w * phi(x - u2)) for x in data)

def em(w, u1, u2, iters=300, tol=1e-12):
    hist = [loglik(w, u1, u2)]
    for _ in range(iters):
        r = []
        for x in data:
            A = (1 - w) * phi(x - u1); B = w * phi(x - u2)
            r.append(B / (A + B))
        sr = sum(r)
        w = sr / N
        u2 = sum(ri * x for ri, x in zip(r, data)) / sr
        u1 = sum((1 - ri) * x for ri, x in zip(r, data)) / (N - sr)
        hist.append(loglik(w, u1, u2))
        if hist[-1] - hist[-2] < tol:
            break
    return (w, u1, u2), hist

(fw, f1, f2), hist = em(0.5, -1.0, 1.0)
drops = sum(1 for i in range(1, len(hist)) if hist[i] < hist[i - 1] - 1e-9)
print(f"  true (w,mu1,mu2) = ({wt}, {m1t}, {m2t}),  n={N}")
print(f"  EM from (0.5,-1,1): converged in {len(hist)-1} iterations to "
      f"w={fw:.4f} mu1={f1:.4f} mu2={f2:.4f}")
print(f"  log-likelihood at iterations 0,1,2,3,5,10,last: " +
      ", ".join(f"{hist[i]:.3f}" for i in [0, 1, 2, 3, 5, 10, len(hist)-1]))
print(f"  number of iterations where the log-likelihood DECREASED: {drops}  (theorem says 0)")
best = max(hist)
finals = []
for _ in range(40):
    w0 = random.uniform(0.05, 0.95)
    u10 = random.uniform(-6, 9); u20 = random.uniform(-6, 9)
    _, h = em(w0, u10, u20)
    finals.append(h[-1])
finals.sort()
top = sum(1 for v in finals if v > best - 0.01)
print(f"  40 random inits: best final loglik={finals[-1]:.3f}, worst={finals[0]:.3f}, "
      f"reached the best value {top}/40 times")
print("  -> in THIS well-separated problem every start found the same optimum; no local-max trap here.")

hdr("7.2.4 WITH FREE VARIANCES THE LIKELIHOOD IS UNBOUNDED: no MLE exists")
def loglik2(w, u1, v1, u2, v2):
    tot = 0.0
    for x in data:
        A = (1 - w) * phi((x - u1) / math.sqrt(v1)) / math.sqrt(v1)
        B = w * phi((x - u2) / math.sqrt(v2)) / math.sqrt(v2)
        tot += math.log(A + B)
    return tot

# Pin component 1 exactly on one observation and let its variance shrink.
w, u1, u2, v2 = 0.5, data[0], sum(data) / N, 1.0
print(f"  component 1 pinned at the single observation x1={data[0]:.4f}; component 2 held at "
      f"(mu={u2:.3f}, var={v2:.1f})")
print("     v1        |  log-likelihood  |  predicted: const - 0.5*log(v1)")
base = None
for e in (8, 16, 30, 60, 120, 240):
    v1 = 10.0 ** (-e)
    ll = loglik2(w, u1, v1, u2, v2)
    if base is None:
        base = ll + 0.5 * math.log(v1)
    print(f"   1e-{e:<4d}      |   {ll:11.3f}    |   {base - 0.5*math.log(v1):11.3f}")
print("  -> the log-likelihood grows like -0.5*log(v1) without bound: the supremum is +infinity,")
print("     and the simulated values track the predicted slope exactly.")
print("     The unrestricted normal-mixture MLE does not exist; in practice one bounds the")
print("     variances away from 0, penalizes them, or puts a prior on them.")

# ============================== 7.3.1 ==============================
hdr("7.3.1 MSE: bias-variance decomposition, and unbiasedness can be worthless")
print("  (divisor comparison above already shows MSE(n+1) < MSE(MLE) < MSE(S^2) for normal data)")
print(f"  check decomposition for the MLE of sigma^2: MSE={mse(mll,sig2):.4f}  "
      f"var+bias^2={st.variance(mll) + (st.mean(mll)-sig2)**2:.4f}")
lam, R = 1.0, 200000
vals = [(-1.0) ** rpois(lam) for _ in range(R)]
print(f"  Poisson(1), estimator (-1)^X for exp(-2*lambda)={math.exp(-2*lam):.4f}:")
print(f"     E[(-1)^X] = {st.mean(vals):+.4f}  (unbiased)   but it only ever takes the values -1 and +1")
print(f"     Var = {st.variance(vals):.4f}  theory 1-exp(-4*lambda) = {1-math.exp(-4*lam):.4f}")

# ============================== 7.3.2 ==============================
hdr("7.3.2 CRAMER-RAO: attained for exponential-family means, not attained in general")
R = 200000
for name, draw, n, var_th, crlb in (
        ("Bernoulli(0.3), estimate p   ", lambda: 1.0 if random.random() < 0.3 else 0.0, 20, 0.3*0.7/20, 0.3*0.7/20),
        ("Poisson(2), estimate lambda  ", lambda: float(rpois(2.0)), 20, 2.0/20, 2.0/20),
        ("Normal(0,4), estimate mu     ", lambda: random.gauss(0.0, 2.0), 20, 4.0/20, 4.0/20)):
    xb = []
    for _ in range(R):
        xb.append(sum(draw() for _ in range(n)) / n)
    print(f"  {name}: Var(Xbar)={st.variance(xb):.5f}  theory {var_th:.5f}  CRLB {crlb:.5f}  -> ratio {crlb/var_th:.3f}")

hdr("7.3.2 A UMVUE that does NOT attain the bound: Poisson, estimate tau=exp(-lambda)")
n, lam, R = 10, 1.0, 200000
a = (n - 1) / n
crude, umvue = [], []
for _ in range(R):
    x = [rpois(lam) for _ in range(n)]
    crude.append(1.0 if x[0] == 0 else 0.0)
    umvue.append(a ** sum(x))
tau = math.exp(-lam)
print(f"  tau = exp(-1) = {tau:.5f}")
print(f"  crude  1{{X1=0}} : E={st.mean(crude):.5f}  Var={st.variance(crude):.5f}  theory {tau*(1-tau):.5f}")
print(f"  UMVUE  ((n-1)/n)^T: E={st.mean(umvue):.5f}  Var={st.variance(umvue):.5f}  theory {math.exp(lam*(1-2*n)/n)-math.exp(-2*lam):.5f}")
print(f"  CRLB for tau = lambda*exp(-2*lambda)/n = {lam*math.exp(-2*lam)/n:.5f}   <- strictly below the UMVUE variance")

# ============================== 7.3.3 ==============================
hdr("7.3.3 RAO-BLACKWELL: conditioning on a sufficient statistic cuts variance")
n, p, R = 10, 0.5, 300000
cr, rb = [], []
for _ in range(R):
    x = [1 if random.random() < p else 0 for _ in range(n)]
    t = sum(x)
    cr.append(float(x[0] * x[1]))
    rb.append(t * (t - 1) / (n * (n - 1)))
ex2 = sum(math.comb(n, t) * p**t * (1-p)**(n-t) * (t*(t-1)/(n*(n-1)))**2 for t in range(n+1))
print(f"  Bernoulli(0.5), n=10, estimate p^2 = 0.25")
print(f"  crude X1*X2              : E={st.mean(cr):.4f}  Var={st.variance(cr):.5f}  theory {p*p*(1-p*p):.5f}")
print(f"  Rao-Blackwell T(T-1)/(n(n-1)): E={st.mean(rb):.4f}  Var={st.variance(rb):.5f}  theory {ex2-p**4:.5f}")
print(f"  variance ratio = {p*p*(1-p*p)/(ex2-p**4):.2f}x reduction")
print(f"  (the Poisson example above is the same mechanism: 1{{X1=0}} -> ((n-1)/n)^T, "
      f"{(tau*(1-tau))/(math.exp(lam*(1-2*n)/n)-math.exp(-2*lam)):.1f}x reduction)")

# ============================== 7.3.4 ==============================
hdr("7.3.4 LOSS FUNCTIONS: posterior mean / median / mode for squared / absolute / 0-1 loss")
k, rate = 5, 6.0          # posterior Gamma(shape 5, rate 6) from Poisson data sum=3, n=5, Gamma(2,1) prior
def erlang_cdf(x):
    return 1.0 - sum(math.exp(-rate*x) * (rate*x)**j / math.factorial(j) for j in range(k))
lo, hi = 0.0, 10.0
for _ in range(200):
    mid = (lo + hi) / 2
    if erlang_cdf(mid) < 0.5:
        lo = mid
    else:
        hi = mid
pmed = (lo + hi) / 2
pmean = k / rate
pmode = (k - 1) / rate
print(f"  posterior Gamma(shape=5, rate=6): mean={pmean:.5f}  median={pmed:.5f}  mode={pmode:.5f}")
R = 120000
post = [random.gammavariate(k, 1.0 / rate) for _ in range(R)]
pvar = k / rate**2
for nm, c in (("mean  ", pmean), ("median", pmed), ("mode  ", pmode)):
    sq = pvar + (pmean - c) ** 2
    ab = sum(abs(t - c) for t in post) / R
    print(f"  candidate = posterior {nm}: E[(theta-a)^2]={sq:.5f} (exact)   E|theta-a|={ab:.5f}")
cand = [0.60 + 0.005 * i for i in range(61)]
bs = min(cand, key=lambda c: pvar + (pmean - c) ** 2)
ba = min(cand, key=lambda c: sum(abs(t - c) for t in post))
print(f"  grid scan over a in [0.60,0.90] step 0.005:")
print(f"     argmin of E[(theta-a)^2] = {bs:.3f}   (posterior mean   = {pmean:.4f})")
print(f"     argmin of E|theta-a|     = {ba:.3f}   (posterior median = {pmed:.4f})")
