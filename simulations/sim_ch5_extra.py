"""Extra Chapter 5 simulations supporting the expanded sections (pure standard library, seeded)."""
import random, math, statistics as st

random.seed(555)

def hdr(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)

# ------------------------------------------------------------------ 5.1
hdr("5.1 Sampling WITHOUT replacement from a finite population (not independent)")
N = 50
pop = [random.gauss(10, 3) for _ in range(N)]
mu = sum(pop) / N
sig2 = sum((x - mu) ** 2 for x in pop) / N
n = 20
R = 100000
wr = [sum(random.choices(pop, k=n)) / n for _ in range(R)]
wo = [sum(random.sample(pop, n)) / n for _ in range(R)]
print(f"  population N={N}, sample n={n}, sigma^2 (divisor N) = {sig2:.4f}")
print(f"  with replacement   : Var(Xbar) = {st.variance(wr):.5f}   theory sigma^2/n = {sig2/n:.5f}")
print(f"  without replacement: Var(Xbar) = {st.variance(wo):.5f}   theory (sigma^2/n)(N-n)/(N-1) = {sig2/n*(N-n)/(N-1):.5f}")

# ------------------------------------------------------------------ 5.2
hdr("5.2 Unbiasedness of S^2 (divisor n-1) vs the divisor-n version; Exp(1) has sigma^2=1, n=5")
R = 200000
s2, b2 = [], []
for _ in range(R):
    xs = [random.expovariate(1.0) for _ in range(5)]
    m = sum(xs) / 5
    ss = sum((x - m) ** 2 for x in xs)
    s2.append(ss / 4); b2.append(ss / 5)
print(f"  E[S^2 (n-1)]     = {st.mean(s2):.4f}   theory 1")
print(f"  E[ss/n]          = {st.mean(b2):.4f}   theory (n-1)/n = 0.8")

hdr("5.2 Cauchy: averaging does NOT help (Xbar_n is again standard Cauchy). P(|Xbar_n|>1) = 1/2 for all n")
R = 4000
for n in (1, 10, 100, 1000):
    cnt = 0
    for _ in range(R):
        s = 0.0
        for _ in range(n):
            s += math.tan(math.pi * (random.random() - 0.5))
        if abs(s / n) > 1:
            cnt += 1
    normal_th = math.erfc(math.sqrt(n) / math.sqrt(2))
    print(f"  n={n:5d}: Cauchy empirical P(|Xbar|>1) = {cnt/R:.4f}  (theory 0.5000)   [for N(0,1) data it would be {normal_th:.4f}]")

# ------------------------------------------------------------------ 5.3
hdr("5.3 chi-square_1 from Z^2, and F -> Beta relation")
R = 200000
z2 = [random.gauss(0, 1) ** 2 for _ in range(R)]
print(f"  Z^2: mean={st.mean(z2):.4f} (1)  var={st.variance(z2):.4f} (2)  P(Z^2<=1)={sum(v<=1 for v in z2)/R:.4f}  (P(|Z|<=1)=0.6827)")
p, q = 3, 10
ys = []
for _ in range(R):
    U = sum(random.gauss(0, 1) ** 2 for _ in range(p))
    V = sum(random.gauss(0, 1) ** 2 for _ in range(q))
    ys.append(U / (U + V))              # = (p/q F)/(1+(p/q)F)
a, b = p / 2, q / 2
print(f"  Y=(p/q)F/(1+(p/q)F), p=3,q=10: mean={st.mean(ys):.4f} (Beta(1.5,5): {a/(a+b):.4f})  var={st.variance(ys):.5f} ({a*b/((a+b)**2*(a+b+1)):.5f})")

hdr("5.3 Coverage of 'xbar +/- c * s/sqrt(n)' for N(0,1) data: z=1.96 versus the t critical value")
tcrit = {3: 4.303, 5: 2.776, 10: 2.262, 30: 2.045}
R = 100000
for n, tc in tcrit.items():
    cz = ct = 0
    for _ in range(R):
        x = [random.gauss(0, 1) for _ in range(n)]
        m = sum(x) / n
        s = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
        h = s / math.sqrt(n)
        cz += abs(m) <= 1.96 * h
        ct += abs(m) <= tc * h
    print(f"  n={n:3d}: coverage with z=1.96: {cz/R:.4f}    with t_(.975,{n-1})={tc}: {ct/R:.4f}   (nominal 0.95)")

# ------------------------------------------------------------------ 5.4
hdr("5.4 Order statistics of Uniform(0,1): range ~ Beta(n-1,2), median(n odd) ~ Beta((n+1)/2,(n+1)/2)")
R = 200000
rng, med = [], []
for _ in range(R):
    x = sorted(random.random() for _ in range(11))      # fresh sample of 11 for the median
    med.append(x[5])
    y = [random.random() for _ in range(10)]             # fresh, independent sample of 10 for the range
    rng.append(max(y) - min(y))
print(f"  range of n=10: mean={st.mean(rng):.4f} (Beta(9,2): {9/11:.4f})  var={st.variance(rng):.5f} ({18/(121*12):.5f})")
print(f"  median of n=11: mean={st.mean(med):.4f} (0.5)  var={st.variance(med):.5f} (Beta(6,6): {36/(144*13):.5f})")

# ------------------------------------------------------------------ 5.5
hdr("5.5 Slutsky in action: coverage of the NOMINAL-95% t interval for the mean of Exp(1) data")
tc = {10: 2.262, 30: 2.045, 100: 1.984}
R = 50000
for n, c in tc.items():
    cov = 0
    for _ in range(R):
        x = [random.expovariate(1.0) for _ in range(n)]
        m = sum(x) / n
        s = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
        cov += abs(m - 1.0) <= c * s / math.sqrt(n)
    print(f"  n={n:4d}: coverage = {cov/R:.4f}")

hdr("5.5 Second-order delta method: X_i~Bernoulli(1/2), g(p)=p(1-p), g'(1/2)=0")
R = 200000
for n in (100, 400):
    vals = []
    for _ in range(R):
        xbar = random.getrandbits(n).bit_count() / n
        vals.append(n * (xbar * (1 - xbar) - 0.25))
    prob = sum(-4 * v <= 3.841 for v in vals) / R        # chi2_1 95th percentile; avoids binomial lattice boundary
    print(f"  n={n:4d}: mean of n(g(Xbar)-1/4) = {st.mean(vals):+.4f} (theory -0.25)   var = {st.variance(vals):.4f} (limit 1/8 = 0.125)   P(-4V<=3.841) = {prob:.4f} (chi2_1: 0.9500)")

# ------------------------------------------------------------------ 5.6
hdr("5.6.2 Box-Muller (direct) and Poisson via exponential inter-arrival times (indirect)")
R = 300000
z1s, z2s = [], []
for _ in range(R):
    u1 = 1.0 - random.random(); u2 = random.random()
    r = math.sqrt(-2.0 * math.log(u1)); t = 2 * math.pi * u2
    z1s.append(r * math.cos(t)); z2s.append(r * math.sin(t))
m1, m2 = st.mean(z1s), st.mean(z2s)
cv = sum((a - m1) * (b - m2) for a, b in zip(z1s, z2s)) / (R - 1)
print(f"  Box-Muller Z1: mean={m1:+.4f} var={st.variance(z1s):.4f}  P(|Z1|<1.96)={sum(abs(v)<1.96 for v in z1s)/R:.4f} (0.9500)  cov(Z1,Z2)={cv:+.4f}")
lam = 3.0
R = 200000
cnts = []
for _ in range(R):
    t = 0.0; k = 0
    while True:
        t += random.expovariate(lam)
        if t > 1.0:
            break
        k += 1
    cnts.append(k)
print(f"  Poisson(3) via arrivals in [0,1]: mean={st.mean(cnts):.4f} var={st.variance(cnts):.4f} (both 3)")
print("   k | empirical pmf | Poisson(3) pmf")
for k in range(0, 6):
    print(f"   {k} |    {cnts.count(k)/R:.4f}     |   {math.exp(-lam)*lam**k/math.factorial(k):.4f}")

hdr("5.6 Monte Carlo error: estimate pi by 4*P(U1^2+U2^2<=1); SE should shrink like 1/sqrt(N)")
sd1 = 4 * math.sqrt((math.pi / 4) * (1 - math.pi / 4))
for N, reps in ((100, 600), (1000, 600), (10000, 400), (100000, 100)):
    ests = []
    for _ in range(reps):
        hit = 0
        for _ in range(N):
            x = random.random(); y = random.random()
            hit += (x * x + y * y <= 1.0)
        ests.append(4.0 * hit / N)
    print(f"  N={N:6d}: mean estimate={st.mean(ests):.4f}   empirical SE={st.stdev(ests):.5f}   theory {sd1/math.sqrt(N):.5f}")
