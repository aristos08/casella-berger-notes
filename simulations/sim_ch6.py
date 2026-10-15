"""Chapter 6 simulations (pure standard library, seeded)."""
import random, math, statistics as st
from itertools import combinations

random.seed(606)

def hdr(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)

# ---------------------------------------------------------------- 6.2.1 (a)
hdr("6.2.1(a) SUFFICIENCY: Bernoulli. Conditional law of X given T=sum is free of p")
n = 4
R = 600000
arrangements = [tuple(1 if i in c else 0 for i in range(n)) for c in combinations(range(n), 2)]
for p in (0.2, 0.7):
    counts = {a: 0 for a in arrangements}
    tot = 0
    for _ in range(R):
        x = tuple(1 if random.random() < p else 0 for _ in range(n))
        if sum(x) == 2:
            counts[x] += 1
            tot += 1
    print(f"p={p}: P(T=2) empirical={tot/R:.4f}  theory={6*p*p*(1-p)**2:.4f};  conditional freq of each of the 6 arrangements (theory 1/6=0.1667):")
    print("   " + "  ".join(f"{''.join(map(str,a))}:{counts[a]/tot:.4f}" for a in arrangements))

# ---------------------------------------------------------------- 6.2.1 (b)
hdr("6.2.1(b) SUFFICIENCY AS REGENERATION: Normal(mu,1), n=11, regenerate data from T")
n = 11
mu = 3.0
R = 120000
def stats(sample):
    return sample[0], max(sample), sample[0] * sample[1]
orig, regen_mean, regen_med = [], [], []
for _ in range(R):
    x = [random.gauss(mu, 1) for _ in range(n)]
    orig.append(stats(x))
    xbar = sum(x) / n
    # regenerate from the sufficient statistic Xbar: X' = Xbar + (Z - Zbar)
    z = [random.gauss(0, 1) for _ in range(n)]
    zbar = sum(z) / n
    xp = [xbar + (zi - zbar) for zi in z]
    regen_mean.append(stats(xp))
    # WRONG regeneration from the median (not sufficient): X'' = med(X) + (Z - med(Z))
    medx = sorted(x)[n // 2]
    medz = sorted(z)[n // 2]
    xpp = [medx + (zi - medz) for zi in z]
    regen_med.append(stats(xpp))
def summarize(name, rows):
    v1 = st.variance([r[0] for r in rows])
    m_max = st.mean([r[1] for r in rows]); s_max = st.stdev([r[1] for r in rows])
    m_prod = st.mean([r[2] for r in rows])
    print(f"  {name:34s} Var(X1)={v1:.4f}  E[max]={m_max:.4f}  sd[max]={s_max:.4f}  E[X1*X2]={m_prod:.4f}")
print(f"  (theory for the true sample: Var(X1)=1, E[X1*X2]=mu^2={mu*mu:.1f})")
summarize("original sample X", orig)
summarize("regenerated from Xbar (sufficient)", regen_mean)
summarize("regenerated from median (NOT suff.)", regen_med)

# ---------------------------------------------------------------- 6.2.3
hdr("6.2.3 ANCILLARITY: Uniform(theta, theta+1), n=5. Range R is ancillary")
n = 5
R = 200000
for theta in (0.0, 10.0, 1000.0):
    rs = []
    for _ in range(R):
        x = [theta + random.random() for _ in range(n)]
        rs.append(max(x) - min(x))
    rs.sort()
    print(f"  theta={theta:7.1f}: E[R]={st.mean(rs):.4f} (theory (n-1)/(n+1)={(n-1)/(n+1):.4f})  "
          f"q10={rs[int(.1*R)]:.4f}  q50={rs[int(.5*R)]:.4f}  q90={rs[int(.9*R)]:.4f}")

hdr("6.2.3 ... but the ancillary R measures PRECISION. Midpoint estimator, conditional on R")
R = 400000
theta = 0.0
rows = []
for _ in range(R):
    x = [theta + random.random() for _ in range(n)]
    lo, hi = min(x), max(x)
    est = (lo + hi) / 2 - 0.5          # midpoint-based estimate of theta
    rows.append((hi - lo, (est - theta) ** 2))
print(f"  overall MSE = {st.mean(r[1] for r in rows):.5f}   theory E[(1-R)^2]/12 = 1/84 = {1/84:.5f}")
bins = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.0)]
print("  R-bin        count    empirical cond. MSE    mean of (1-R)^2/12")
for lo, hi in bins:
    sel = [r for r in rows if lo <= r[0] < hi]
    if sel:
        print(f"  [{lo:.1f},{hi:.1f})   {len(sel):7d}      {st.mean(s[1] for s in sel):.5f}            {st.mean((1-s[0])**2/12 for s in sel):.5f}")

# ---------------------------------------------------------------- 6.2.4
hdr("6.2.4 NOT COMPLETE: g(T)=R-(n-1)/(n+1) has mean 0 for every theta, yet is not 0")
R = 300000
for theta in (-3.0, 0.0, 2.5, 100.0):
    vals = []
    for _ in range(R):
        x = [theta + random.random() for _ in range(n)]
        vals.append(max(x) - min(x) - (n - 1) / (n + 1))
    print(f"  theta={theta:7.1f}: E[g(T)] = {st.mean(vals):+.5f}   sd[g(T)] = {st.stdev(vals):.4f}  (nonzero spread)")

hdr("6.2.4 BASU: Exponential(scale theta), n=5. T=sum is complete sufficient; W=X1/T ancillary")
n = 5
R = 150000
def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    return num / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
for theta in (1.0, 5.0):
    T, W, X1 = [], [], []
    for _ in range(R):
        x = [random.expovariate(1.0 / theta) for _ in range(n)]
        t = sum(x)
        T.append(t); W.append(x[0] / t); X1.append(x[0])
    qs = sorted(T)
    cuts = [qs[int(R * k / 4)] for k in (1, 2, 3)]
    def quart(t):
        return sum(t > c for c in cuts)
    cw = [[], [], [], []]; cx = [[], [], [], []]
    for t, w, x1 in zip(T, W, X1):
        cw[quart(t)].append(w); cx[quart(t)].append(x1)
    print(f"  theta={theta}: E[W]={st.mean(W):.4f} (Beta(1,4): 0.2000)  Var[W]={st.variance(W):.5f} (0.02667)")
    print(f"     corr(W,T)={corr(W,T):+.4f}   corr(W^2,log T)={corr([w*w for w in W],[math.log(t) for t in T]):+.4f}")
    print(f"     E[W  | T quartile 1..4] = " + "  ".join(f"{st.mean(c):.4f}" for c in cw) + "   <- flat: independent of T")
    print(f"     E[X1 | T quartile 1..4] = " + "  ".join(f"{st.mean(c):.4f}" for c in cx) + "   <- NOT flat: X1 is not ancillary")

# ---------------------------------------------------------------- 6.3
hdr("6.3 LIKELIHOOD PRINCIPLE: 9 successes, 3 failures; H0: p=1/2 vs H1: p>1/2")
p_bin = sum(math.comb(12, k) for k in range(9, 13)) / 2 ** 12
p_nb = sum(math.comb(11, k) for k in range(9, 12)) / 2 ** 11
print(f"  Design A (n=12 fixed, Binomial):            one-sided p-value = {p_bin:.5f}")
print(f"  Design B (sample until 3rd failure, NegBin): one-sided p-value = {p_nb:.5f}")
print("  Likelihood ratio L_bin(p)/L_nb(p) at several p (should be the constant C(12,9)/C(11,2)=4):")
for p in (0.1, 0.3, 0.5, 0.75, 0.9):
    lb = math.comb(12, 9) * p ** 9 * (1 - p) ** 3
    ln = math.comb(11, 2) * p ** 9 * (1 - p) ** 3
    print(f"     p={p:4.2f}:  L_bin={lb:.6f}  L_nb={ln:.6f}  ratio={lb/ln:.4f}")

hdr("6.3 STOPPING RULES: type I error of a nominal 5% z-test under 'peeking' (H0 true, N(0,1) data)")
R = 40000
K = 100
rej_10 = 0; rej_all = 0; rej_single = 0
for _ in range(R):
    s = 0.0
    any10 = anyall = False
    for i in range(1, K + 1):
        s += random.gauss(0, 1)
        if abs(s) > 1.96 * math.sqrt(i):
            anyall = True
            if i % 10 == 0:
                any10 = True
    rej_all += anyall
    rej_10 += any10
    rej_single += abs(s) > 1.96 * math.sqrt(K)
print(f"  one look at n=100:               P(reject) = {rej_single/R:.4f}")
print(f"  10 equally spaced looks (10..100): P(reject) = {rej_10/R:.4f}")
print(f"  100 looks (after every obs):       P(reject) = {rej_all/R:.4f}")

# ---------------------------------------------------------------- 6.4
hdr("6.4 EQUIVARIANCE: N(theta,1), n=10. MSE of equivariant vs non-equivariant estimators")
n = 10
R = 120000
print("  theta   MSE[mean]   MSE[median]   MSE[0.8*mean] (not equivariant)   theory MSE[0.8*mean]=0.064+(0.2*theta)^2")
for theta in (0.0, 1.0, 5.0):
    em, emed, esh = [], [], []
    for _ in range(R):
        x = [random.gauss(theta, 1) for _ in range(n)]
        m = sum(x) / n
        sx = sorted(x)
        med = (sx[n // 2 - 1] + sx[n // 2]) / 2
        em.append((m - theta) ** 2); emed.append((med - theta) ** 2); esh.append((0.8 * m - theta) ** 2)
    th = 0.64 * (1 / n) + (0.2 * theta) ** 2
    print(f"  {theta:5.1f}   {st.mean(em):.4f}      {st.mean(emed):.4f}        {st.mean(esh):.4f}                          {th:.4f}")
