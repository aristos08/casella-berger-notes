import random, math, statistics as st

random.seed(2024)

def percentile(sorted_xs, q):
    n = len(sorted_xs)
    idx = int(q*n)
    return sorted_xs[min(idx, n-1)]

def beta_fn(a,b):
    return math.gamma(a)*math.gamma(b)/math.gamma(a+b)

def beta_pdf(x,a,b):
    if x<=0 or x>=1: return 0.0
    return x**(a-1)*(1-x)**(b-1)/beta_fn(a,b)

def norm_pdf(x, mu=0, sigma=1):
    return math.exp(-0.5*((x-mu)/sigma)**2)/(sigma*math.sqrt(2*math.pi))

print("="*70)
print("5.4  ORDER STATISTICS")
print("="*70)

n, j = 10, 3   # look at X_(3) out of 10 uniforms
N = 300000
samples_j = []
for _ in range(N):
    s = sorted(random.random() for _ in range(n))
    samples_j.append(s[j-1])  # 0-indexed -> j-th order stat

a, b = j, n-j+1
emp_mean, emp_var = st.mean(samples_j), st.variance(samples_j)
th_mean = a/(a+b)
th_var = a*b/((a+b)**2*(a+b+1))
print(f"\nX_({j}) out of n={n} iid Uniform(0,1), N={N} reps.")
print(f"Theory: X_({j}) ~ Beta({a},{b})")
print(f"  empirical mean = {emp_mean:.4f}   theory = {th_mean:.4f}")
print(f"  empirical var  = {emp_var:.5f}   theory = {th_var:.5f}")

bins = [i/20 for i in range(21)]
counts = [0]*20
for x in samples_j:
    k = min(int(x*20), 19)
    counts[k]+=1
print("  bin midpoint | empirical density | Beta(3,8) pdf")
for i,c in enumerate(counts):
    mid = (bins[i]+bins[i+1])/2
    emp_dens = c/N/0.05
    print(f"    {mid:4.2f}      |      {emp_dens:6.4f}       |   {beta_pdf(mid,a,b):6.4f}")

# max and min as special cases
maxs = [max(random.random() for _ in range(n)) for _ in range(100000)]
mins = [min(random.random() for _ in range(n)) for _ in range(100000)]
print(f"\nSanity: X_(n) [max] ~ Beta(n,1): empirical mean={st.mean(maxs):.4f}, theory n/(n+1)={n/(n+1):.4f}")
print(f"Sanity: X_(1) [min] ~ Beta(1,n): empirical mean={st.mean(mins):.4f}, theory 1/(n+1)={1/(n+1):.4f}")

print("\n" + "="*70)
print("5.5.1  CONVERGENCE IN PROBABILITY (WLLN)")
print("="*70)

mu_exp = 1.0
eps = 0.1
print(f"\nX_i iid Exp(rate=1) [mu=1]. Tracking P(|Xbar_n - 1| > {eps}) as n grows:")
for n_ in [10, 50, 200, 1000, 5000]:
    Ntrial = 20000
    bad = 0
    for _ in range(Ntrial):
        xbar = sum(random.expovariate(1.0) for _ in range(n_))/n_
        if abs(xbar-mu_exp) > eps:
            bad += 1
    print(f"  n={n_:5d}:  empirical P(|Xbar-mu|>{eps}) = {bad/Ntrial:.4f}")

print("\n" + "="*70)
print("5.5.2  ALMOST SURE CONVERGENCE: counterexample (sweeping/typewriter sequence)")
print("="*70)
print("""
Construct deterministic sets on ([0,1], Lebesgue measure), indexed by (m,k):
  n = m(m-1)/2 + k,   A_{m,k} = [(k-1)/m, k/m),   k=1,...,m,  m=1,2,...
  X_n(w) = 1_{A_n}(w).

As n -> infinity, P(A_n) = 1/m -> 0 along each sweep, so X_n -> 0 in probability.
But for EVERY fixed w in [0,1), each sweep m has exactly one k with w in A_{m,k},
so X_n(w)=1 infinitely often -- X_n(w) never settles down. No a.s. convergence.
""")

def n_of(m,k):
    return m*(m-1)//2 + k

w = 0.37
hits = []
for m in range(1,8):
    for k in range(1,m+1):
        lo, hi = (k-1)/m, k/m
        if lo <= w < hi:
            hits.append((n_of(m,k), m, 1/m))
print(f"Fixed point w={w}. Sweeps m=1..7, the n at which X_n(w)=1, and P(A_n)=1/m at that sweep:")
for n_, m, p in hits:
    print(f"  sweep m={m:2d}:  n={n_:3d}   X_n({w})=1   while P(A_n)=1/m = {p:.4f}")
print("-> X_n(w) keeps hitting 1 forever (never converges pointwise to 0),")
print("   even though P(A_n) -> 0, i.e. convergence in probability still holds.")

print("\n" + "="*70)
print("5.5.3  CONVERGENCE IN DISTRIBUTION: CLT via standardized sums")
print("="*70)

rate = 1.0
mu_, sigma_ = 1.0/rate, 1.0/rate  # Exp(1) mean=1, sd=1
for n_ in [2, 10, 50, 200]:
    N = 300000
    Zs = []
    for _ in range(N):
        xbar = sum(random.expovariate(rate) for _ in range(n_))/n_
        Zs.append((xbar - mu_)/(sigma_/math.sqrt(n_)))
    print(f"\nn={n_}: standardized Xbar from Exp(1), N={N} reps")
    print(f"  mean={st.mean(Zs): .4f} (theory 0)   var={st.variance(Zs): .4f} (theory 1)")
    Zs_sorted = sorted(Zs)
    print(f"  skewness proxy: P(Z<-1)={sum(1 for z in Zs if z<-1)/N:.4f} vs Phi(-1)=0.1587"
          f" | P(Z>1)={sum(1 for z in Zs if z>1)/N:.4f} vs 1-Phi(1)=0.1587")

print("\n" + "="*70)
print("5.5.4  THE DELTA METHOD")
print("="*70)

p = 0.3
def logit(x): return math.log(x/(1-x))
gprime_p = 1/(p*(1-p))
theory_var = p*(1-p)*gprime_p**2   # = 1/(p(1-p))
print(f"\nX_i iid Bernoulli(p={p}), g(x)=logit(x)=log(x/(1-x)), g'(p)={gprime_p:.4f}")
print(f"Delta method theory: sqrt(n)(g(Xbar)-g(p)) -> N(0, sigma^2) with sigma^2={theory_var:.4f}")
for n_ in [50, 500, 5000]:
    N = 200000
    vals = []
    for _ in range(N):
        xbar = sum(1 for _ in range(n_) if random.random()<p)/n_
        # guard against xbar in {0,1} for logit
        if xbar <= 0: xbar = 0.5/n_
        if xbar >= 1: xbar = 1 - 0.5/n_
        vals.append(math.sqrt(n_)*(logit(xbar)-logit(p)))
    print(f"  n={n_:5d}: empirical var = {st.variance(vals):.4f}   theory = {theory_var:.4f}")

print("\n" + "="*70)
print("5.6.1  DIRECT METHODS: Inverse CDF")
print("="*70)

lam = 2.0
N = 300000
# F(x) = 1-exp(-lam x) => F^{-1}(u) = -ln(1-u)/lam
exp_samples = [-math.log(1-random.random())/lam for _ in range(N)]
print(f"\nGenerate Exp(rate={lam}) via X=-ln(1-U)/lam, U~Unif(0,1), N={N}:")
print(f"  empirical mean = {st.mean(exp_samples):.4f}   theory 1/lam = {1/lam:.4f}")
print(f"  empirical var  = {st.variance(exp_samples):.4f}   theory 1/lam^2 = {1/lam**2:.4f}")

print("\n" + "="*70)
print("5.6.3  ACCEPT/REJECT ALGORITHM")
print("="*70)

a2, b2 = 2, 3
# find M = max of beta pdf over (0,1) via fine grid search (mode at (a-1)/(a+b-2))
mode = (a2-1)/(a2+b2-2)
M = beta_pdf(mode, a2, b2)
print(f"\nTarget f = Beta({a2},{b2}) pdf. Proposal g = Uniform(0,1) pdf (=1 on (0,1)).")
print(f"M = sup f/g = max of Beta pdf = f({mode:.3f}) = {M:.4f}")

N = 400000
accepted = []
n_proposed = 0
while len(accepted) < N:
    n_proposed += 1
    Y = random.random()
    U = random.random()
    if U <= beta_pdf(Y,a2,b2)/M:
        accepted.append(Y)

print(f"Proposed {n_proposed} candidates to get {N} accepted samples.")
print(f"  empirical acceptance rate = {N/n_proposed:.4f}   theory 1/M = {1/M:.4f}")
emp_mean = st.mean(accepted)
th_mean = a2/(a2+b2)
print(f"  empirical mean = {emp_mean:.4f}   theory a/(a+b) = {th_mean:.4f}")

bins = [i/20 for i in range(21)]
counts = [0]*20
for x in accepted:
    k = min(int(x*20), 19)
    counts[k]+=1
print("  bin midpoint | empirical density | Beta(2,3) pdf")
for i,c in enumerate(counts):
    mid = (bins[i]+bins[i+1])/2
    emp_dens = c/N/0.05
    print(f"    {mid:4.2f}      |      {emp_dens:6.4f}       |   {beta_pdf(mid,a2,b2):6.4f}")
