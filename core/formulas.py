import sympy as sp
from math import comb
from scipy.stats import norm, binom, poisson, hypergeom

FORMULAS = {
    # ---------------- Binomial ----------------
    "binomial_mean": {
        "keywords": ["binomial", "mean", "expected"],
        "latex": r"\mathbb{E}[X] = np",
        "params": ["n", "p"],
        "func": lambda n, p: n * p
    },
    "binomial_variance": {
        "keywords": ["binomial", "variance"],
        "latex": r"\mathrm{Var}[X] = np(1-p)",
        "params": ["n", "p"],
        "func": lambda n, p: n * p * (1 - p)
    },
    "binomial_pmf": {
        "keywords": ["binomial", "pmf", "probability", "exact"],
        "latex": r"P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}",
        "params": ["n", "p", "k"],
        "func": lambda n, p, k: binom.pmf(k, n, p)
    },
    "binomial_cdf": {
        "keywords": ["binomial", "cdf", "cumulative"],
        "latex": r"P(X \leq k) = \sum_{i=0}^k \binom{n}{i} p^i (1-p)^{n-i}",
        "params": ["n", "p", "k"],
        "func": lambda n, p, k: binom.cdf(k, n, p)
    },

    # ---------------- Hypergeometric ----------------
    "hypergeom_mean": {
        "keywords": ["hypergeometric", "mean"],
        "latex": r"\mathbb{E}[X] = n \cdot \frac{K}{N}",
        "params": ["N", "K", "n"],
        "func": lambda N, K, n: n * (K / N)
    },
    "hypergeom_variance": {
        "keywords": ["hypergeometric", "variance"],
        "latex": r"\mathrm{Var}[X] = n \cdot \frac{K}{N} \cdot \frac{N-K}{N} \cdot \frac{N-n}{N-1}",
        "params": ["N", "K", "n"],
        "func": lambda N, K, n: n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1))
    },

    # ---------------- Poisson ----------------
    "poisson_mean": {
        "keywords": ["poisson", "mean", "expected"],
        "latex": r"\mathbb{E}[X] = \lambda",
        "params": ["lmbda"],
        "func": lambda lmbda: lmbda
    },
    "poisson_variance": {
        "keywords": ["poisson", "variance"],
        "latex": r"\mathrm{Var}[X] = \lambda",
        "params": ["lmbda"],
        "func": lambda lmbda: lmbda
    },
    "poisson_pmf": {
        "keywords": ["poisson", "pmf", "probability"],
        "latex": r"P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}",
        "params": ["lmbda", "k"],
        "func": lambda lmbda, k: poisson.pmf(k, lmbda)
    },

    # ---------------- Normal ----------------
    "normal_pdf": {
        "keywords": ["normal", "pdf", "density"],
        "latex": r"f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
        "params": ["x", "mu", "sigma"],
        "func": lambda x, mu, sigma: norm.pdf(x, mu, sigma)
    },
    "normal_cdf": {
        "keywords": ["normal", "cdf", "probability"],
        "latex": r"\Phi(z) = P(Z \leq z)",
        "params": ["z"],
        "func": lambda z: norm.cdf(z)
    },
    "z_score": {
        "keywords": ["z", "z-score", "standardize"],
        "latex": r"z = \frac{x-\mu}{\sigma}",
        "params": ["x", "mu", "sigma"],
        "func": lambda x, mu, sigma: (x - mu) / sigma
    },

    # ---------------- Inequalities ----------------
    "chebyshev_bound": {
        "keywords": ["chebyshev", "inequality", "bound"],
        "latex": r"P(|X-\mu|\geq k\sigma) \leq \frac{1}{k^2}",
        "params": ["k"],
        "func": lambda k: 1 / (k**2)
    },
    "markov_bound": {
        "keywords": ["markov", "inequality", "bound"],
        "latex": r"P(X \geq a) \leq \frac{\mathbb{E}[X]}{a}",
        "params": ["mu", "a"],
        "func": lambda mu, a: mu / a
    }
}
