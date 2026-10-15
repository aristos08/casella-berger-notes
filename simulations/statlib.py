"""Small pure-standard-library distribution helpers shared by the simulation scripts.

Only math/random are used. The incomplete-beta continued fraction is the standard
Lentz algorithm; it gives exact Student-t and Snedecor-F tail probabilities, which
are needed for p-values in Chapters 8, 10 and 11.
"""
import math

# ---------------------------------------------------------------- normal
def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def norm_pdf(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

def norm_quantile(p, lo=-40.0, hi=40.0):
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# ---------------------------------------------------------------- incomplete beta
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h

def betainc(a, b, x):
    """Regularized incomplete beta I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b

# ---------------------------------------------------------------- Student t
def t_cdf(t, nu):
    x = nu / (nu + t * t)
    half = 0.5 * betainc(nu / 2.0, 0.5, x)
    return 1.0 - half if t > 0 else half

def t_sf_two_sided(t, nu):
    """P(|T| >= |t|): the exact two-sided p-value."""
    return betainc(nu / 2.0, 0.5, nu / (nu + t * t))

def t_quantile(p, nu, lo=-200.0, hi=200.0):
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# ---------------------------------------------------------------- Snedecor F
def f_cdf(f, d1, d2):
    if f <= 0:
        return 0.0
    return betainc(d1 / 2.0, d2 / 2.0, d1 * f / (d1 * f + d2))

def f_quantile(p, d1, d2, lo=0.0, hi=1e6):
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if f_cdf(mid, d1, d2) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def beta_quantile(p, a, b, lo=0.0, hi=1.0):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# ---------------------------------------------------------------- incomplete gamma
def _gser(a, x):
    ap, s, d = a, 1.0 / a, 1.0 / a
    for _ in range(1000):
        ap += 1.0
        d *= x / ap
        s += d
        if abs(d) < abs(s) * 1e-16:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))

def _gcf(a, x):
    FPMIN, EPS = 1e-300, 1e-16
    b = x + 1.0 - a
    c = 1.0 / FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < FPMIN:
            d = FPMIN
        c = b + an / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))

def gammainc_P(a, x):
    """Regularized lower incomplete gamma P(a,x)."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        return _gser(a, x)
    return 1.0 - _gcf(a, x)

# ---------------------------------------------------------------- chi-square / gamma
def chi2_cdf(x, k):
    return gammainc_P(k / 2.0, x / 2.0)

def chi2_quantile(p, k, lo=0.0, hi=1e6):
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, k) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def gamma_cdf(x, shape, rate):
    return gammainc_P(shape, rate * x)

def gamma_pdf(x, shape, rate):
    if x <= 0:
        return 0.0
    return math.exp(shape * math.log(rate) + (shape - 1) * math.log(x)
                    - rate * x - math.lgamma(shape))

def gamma_quantile(p, shape, rate, lo=0.0, hi=1e4):
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if gamma_cdf(mid, shape, rate) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# ---------------------------------------------------------------- binomial
def binom_pmf(k, n, p):
    return math.comb(n, k) * p ** k * (1.0 - p) ** (n - k)

def binom_sf(k, n, p):
    """P(X >= k)."""
    return sum(binom_pmf(j, n, p) for j in range(k, n + 1))

if __name__ == "__main__":
    # self-checks against values quoted in standard tables
    print("t_quantile(0.975, 9)  =", round(t_quantile(0.975, 9), 4), " table 2.2622")
    print("t_quantile(0.975, 29) =", round(t_quantile(0.975, 29), 4), " table 2.0452")
    print("two-sided p at t=2.2622, nu=9 =", round(t_sf_two_sided(2.2622, 9), 6), " expect 0.05")
    print("f_quantile(0.95, 3, 10) =", round(f_quantile(0.95, 3, 10), 4), " table 3.7083")
    print("norm_quantile(0.975) =", round(norm_quantile(0.975), 5), " expect 1.95996")
    print("chi2_quantile(0.975, 9) =", round(chi2_quantile(0.975, 9), 4), " table 19.0228")
    print("chi2_quantile(0.025, 9) =", round(chi2_quantile(0.025, 9), 4), " table 2.7004")
    print("chi2_quantile(0.95, 1)  =", round(chi2_quantile(0.95, 1), 4), " table 3.8415")
    print("beta_quantile(0.5, 1, 1) =", round(beta_quantile(0.5, 1, 1), 6), " expect 0.5")
    print("gamma_cdf(1, 1, 1) =", round(gamma_cdf(1, 1, 1), 6), " expect 1-exp(-1)=0.632121")
