import random, math, statistics as st

random.seed(42)

def pearson_corr(xs, ys):
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys)) / (n-1)
    sx, sy = st.pstdev(xs), st.pstdev(ys)
    # use population sd consistent with cov normalization roughly; fine for diagnostic purposes
    sx = math.sqrt(sum((x-mx)**2 for x in xs)/(n-1))
    sy = math.sqrt(sum((y-my)**2 for y in ys)/(n-1))
    return cov/(sx*sy)

def percentile(sorted_xs, q):
    n = len(sorted_xs)
    idx = int(q*n)
    return sorted_xs[min(idx, n-1)]

print("="*70)
print("5.2  SUMS OF IID RANDOM VARIABLES")
print("="*70)

# --- Sum of iid Exponential(rate=1) -> Gamma(n,1) ---
n = 5
N = 200000
sums = [sum(random.expovariate(1.0) for _ in range(n)) for _ in range(N)]
print(f"\nSum of {n} iid Exp(rate=1), N={N} reps:")
print(f"  empirical mean = {st.mean(sums):.4f}   theory (Gamma(n,1) mean=n)      = {n}")
print(f"  empirical var  = {st.variance(sums):.4f}   theory (Gamma(n,1) var=n)       = {n}")

# compare empirical density at a few points to Gamma(n,1) pdf via histogram
def gamma_pdf(x, k, theta=1.0):
    return x**(k-1)*math.exp(-x/theta)/(theta**k*math.gamma(k))

bins = [(i, i+0.5) for i in range(0,12)]
counts = [0]*len(bins)
for s in sums:
    b = int(s/0.5)
    if 0 <= b < len(bins):
        counts[b]+=1
print("  bin midpoint | empirical density | Gamma(5,1) pdf")
for (lo,hi),c in zip(bins,counts):
    mid = (lo+hi)/2*0.5 + 0.25  # not used, recompute properly below
for i,c in enumerate(counts):
    lo, hi = i*0.5, (i+1)*0.5
    mid = (lo+hi)/2
    emp_dens = c/N/0.5
    print(f"    {mid:5.2f}       |     {emp_dens:6.4f}       |   {gamma_pdf(mid,n):6.4f}")

# --- Sum of iid Bernoulli(p) -> Binomial(n,p) ---
p, nb = 0.3, 20
Nb = 300000
sums_b = [sum(1 for _ in range(nb) if random.random()<p) for _ in range(Nb)]
print(f"\nSum of {nb} iid Bernoulli({p}), N={Nb} reps:")
emp_pmf = [sums_b.count(k)/Nb for k in range(nb+1)]
def binom_pmf(k,n,p):
    return math.comb(n,k)*p**k*(1-p)**(n-k)
print("  k | empirical pmf | Binomial(20,0.3) pmf")
for k in range(3,10):
    print(f"  {k:2d} |   {emp_pmf[k]:.4f}     |   {binom_pmf(k,nb,p):.4f}")

print("\n" + "="*70)
print("5.3.1  SAMPLE MEAN & VARIANCE: INDEPENDENCE (NORMAL vs NON-NORMAL)")
print("="*70)

n = 10
N = 300000
mu, sigma = 5.0, 2.0

xbars_norm, s2_norm = [], []
for _ in range(N):
    sample = [random.gauss(mu, sigma) for _ in range(n)]
    xbars_norm.append(st.mean(sample))
    s2_norm.append(st.variance(sample))

corr_norm = pearson_corr(xbars_norm, s2_norm)
print(f"\nNormal(mu={mu}, sigma={sigma}), n={n}, N={N} reps:")
print(f"  corr(Xbar, S^2) = {corr_norm:.4f}   (theory: exactly 0)")

scaled = [(n-1)*s2/sigma**2 for s2 in s2_norm]
print(f"  mean[(n-1)S^2/sigma^2] = {st.mean(scaled):.3f}   theory (chi2_{n-1} mean = {n-1}) = {n-1}")
print(f"  var [(n-1)S^2/sigma^2] = {st.variance(scaled):.3f}   theory (chi2_{n-1} var  = 2(n-1)) = {2*(n-1)}")
print(f"  var[Xbar] empirical = {st.variance(xbars_norm):.4f}   theory sigma^2/n = {sigma**2/n:.4f}")

rate = 1.0
xbars_exp, s2_exp = [], []
for _ in range(N):
    sample = [random.expovariate(rate) for _ in range(n)]
    xbars_exp.append(st.mean(sample))
    s2_exp.append(st.variance(sample))
corr_exp = pearson_corr(xbars_exp, s2_exp)
print(f"\nExponential(rate={rate}), n={n}, N={N} reps:")
print(f"  corr(Xbar, S^2) = {corr_exp:.4f}   (theory: nonzero -- normality is essential!)")

print("\n" + "="*70)
print("5.3.2  STUDENT'S t AND SNEDECOR'S F")
print("="*70)

n = 10
df = n-1
N = 400000
Ts = []
for _ in range(N):
    sample = [random.gauss(mu, sigma) for _ in range(n)]
    xbar = st.mean(sample)
    s = st.stdev(sample)
    Ts.append((xbar-mu)/(s/math.sqrt(n)))

print(f"\nT = (Xbar-mu)/(S/sqrt(n)), n={n} (df={df}), N={N} reps:")
print(f"  empirical mean      = {st.mean(Ts):.4f}   theory = 0")
print(f"  empirical var       = {st.variance(Ts):.4f}   theory df/(df-2) = {df/(df-2):.4f}")
Ts_sorted = sorted(Ts)
print(f"  empirical 97.5th pct = {percentile(Ts_sorted,0.975):.4f}   known table value t_(0.975,{df}) = 2.262")
print(f"  empirical 2.5th  pct = {percentile(Ts_sorted,0.025):.4f}   known table value -t_(0.975,{df}) = -2.262")

# chi-square via sum of squared standard normals
def chisq(k):
    return sum(random.gauss(0,1)**2 for _ in range(k))

p_, q_ = 3, 10
N2 = 300000
Fs = []
for _ in range(N2):
    U = chisq(p_)
    V = chisq(q_)
    Fs.append((U/p_)/(V/q_))
print(f"\nF = (U/p)/(V/q), U~chi2_{p_}, V~chi2_{q_}, N={N2} reps:")
print(f"  empirical mean = {st.mean(Fs):.4f}   theory q/(q-2) = {q_/(q_-2):.4f}")

# verify T^2 ~ F_{1,df} in law (compare moments)
T2 = [t*t for t in Ts]
F1df = []
for _ in range(N2):
    U = chisq(1)
    V = chisq(df)
    F1df.append((U/1)/(V/df))
print(f"\nChecking T^2 =_d F_(1,{df}):")
print(f"  mean(T^2)    = {st.mean(T2):.4f}     mean(F_1,{df}) direct sim = {st.mean(F1df):.4f}   (theory both = {df/(df-2):.4f})")
print(f"  var(T^2)     = {st.variance(T2):.4f}     var(F_1,{df}) direct sim = {st.variance(F1df):.4f}")
