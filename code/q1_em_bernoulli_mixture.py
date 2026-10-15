"""
Assignment 2, Question 1(i)
------------------------------------------------------------------
Expectation-Maximization for a Mixture of Bernoulli (Bernoulli Mixture
Model, BMM) distributions, fitted from scratch (only numpy is used for
basic linear algebra / array operations -- the EM algorithm itself is
implemented manually, no sklearn / scipy.stats mixture routines).

Data: A2Q1.csv -- 400 points in {0,1}^50 (each row is a flattened
10x5 binary matrix). Since every coordinate is binary, the natural
generative model is a MIXTURE OF (MULTIVARIATE, INDEPENDENT) BERNOULLI
DISTRIBUTIONS, not a Gaussian mixture (Gaussians are for continuous
data).

Model
-----
    p(x) = sum_{k=1}^{K} pi_k * prod_{d=1}^{D} mu_{kd}^{x_d} (1-mu_{kd})^{1-x_d}

Parameters: pi (K,), mu (K,D).

This script:
  1. Runs EM for K = 4 with 100 random initializations.
  2. Records the (incomplete-data) log-likelihood at every iteration
     for every initialization.
  3. Plots the log-likelihood averaged over the 100 initializations as
     a function of iteration number.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 42
K = 4
N_ITERS = 50          # number of EM iterations to run / plot
N_INITS = 100          # number of random restarts to average over
EPS = 1e-10             # numerical floor to avoid log(0) / div-by-0


def load_data(path):
    X = np.loadtxt(path, delimiter=",")
    return X


def init_params(D, K, rng):
    """Random initialization of mixing weights and Bernoulli means.

    mu is initialized away from the {0,1} boundary (e.g. Uniform(0.25,0.75))
    so that no probability starts at exactly 0 or 1 (which would make the
    likelihood of some data point exactly zero and stall EM).
    """
    pi = rng.dirichlet(np.ones(K))
    mu = rng.uniform(0.25, 0.75, size=(K, D))
    return pi, mu


def log_bernoulli_pdf(X, mu):
    """
    Returns an (N, K) matrix of log p(x_n | component k) for independent
    Bernoulli coordinates:
        log p(x_n|k) = sum_d [ x_nd log mu_kd + (1-x_nd) log(1-mu_kd) ]
    computed with matrix algebra for speed:
        = X @ log(mu).T + (1-X) @ log(1-mu).T
    """
    mu = np.clip(mu, EPS, 1 - EPS)
    log_mu = np.log(mu)          # (K, D)
    log_1m_mu = np.log(1 - mu)   # (K, D)
    return X @ log_mu.T + (1 - X) @ log_1m_mu.T  # (N, K)


def e_step(X, pi, mu):
    """Compute responsibilities gamma_{nk} and the log-likelihood."""
    log_p_xk = log_bernoulli_pdf(X, mu)          # (N,K) = log p(x_n|k)
    log_pi = np.log(np.clip(pi, EPS, None))       # (K,)
    log_joint = log_p_xk + log_pi[None, :]        # log( pi_k p(x_n|k) )

    # log-sum-exp for numerical stability
    m = np.max(log_joint, axis=1, keepdims=True)
    log_norm = m + np.log(np.sum(np.exp(log_joint - m), axis=1, keepdims=True))
    log_resp = log_joint - log_norm
    resp = np.exp(log_resp)                       # (N,K) responsibilities

    log_likelihood = np.sum(log_norm)
    return resp, log_likelihood


def m_step(X, resp):
    """Update pi and mu given responsibilities."""
    N_k = resp.sum(axis=0)                 # (K,)
    N_k_safe = np.clip(N_k, EPS, None)
    mu = (resp.T @ X) / N_k_safe[:, None]  # (K,D)
    pi = N_k / X.shape[0]
    return pi, mu


def run_em(X, K, n_iters, rng):
    N, D = X.shape
    pi, mu = init_params(D, K, rng)
    ll_trace = np.zeros(n_iters)
    for t in range(n_iters):
        resp, ll = e_step(X, pi, mu)
        ll_trace[t] = ll
        pi, mu = m_step(X, resp)
    return ll_trace, pi, mu


def main():
    X = load_data("/home/claude/assignment2/data/A2Q1.csv")
    N, D = X.shape
    print(f"Loaded data: N={N}, D={D}")

    master_rng = np.random.default_rng(RNG_SEED)
    all_traces = np.zeros((N_INITS, N_ITERS))
    final_params = []

    for init_idx in range(N_INITS):
        rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1))
        ll_trace, pi, mu = run_em(X, K, N_ITERS, rng)
        all_traces[init_idx] = ll_trace
        final_params.append((ll_trace[-1], pi, mu))
        if (init_idx + 1) % 20 == 0:
            print(f"  init {init_idx+1}/{N_INITS} done, final LL = {ll_trace[-1]:.3f}")

    avg_ll = all_traces.mean(axis=0)
    std_ll = all_traces.std(axis=0)

    # Save average trace
    np.savetxt("/home/claude/assignment2/plots/q1_em_avg_loglik.csv",
               np.column_stack([np.arange(1, N_ITERS + 1), avg_ll, std_ll]),
               delimiter=",", header="iteration,avg_loglik,std_loglik", comments="")

    # Plot
    plt.figure(figsize=(7, 5))
    iters = np.arange(1, N_ITERS + 1)
    plt.plot(iters, avg_ll, color="#1f77b4", linewidth=2,
              label="Average log-likelihood (100 random inits)")
    plt.fill_between(iters, avg_ll - std_ll, avg_ll + std_ll, alpha=0.2, color="#1f77b4",
                      label="±1 std across inits")
    plt.xlabel("EM Iteration")
    plt.ylabel("Log-likelihood (averaged over 100 random initializations)")
    plt.title("EM for Bernoulli Mixture Model (K=4) on A2Q1.csv")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/claude/assignment2/plots/q1_em_loglikelihood.png", dpi=150)
    plt.close()

    # Report best run's final mixing weights / cluster sizes
    best = max(final_params, key=lambda t: t[0])
    best_ll, best_pi, best_mu = best
    print(f"\nBest final log-likelihood across {N_INITS} inits: {best_ll:.3f}")
    print(f"Best run mixing weights pi: {np.round(best_pi, 3)}")

    with open("/home/claude/assignment2/plots/q1_em_summary.txt", "w") as f:
        f.write(f"N_INITS={N_INITS}, N_ITERS={N_ITERS}, K={K}\n")
        f.write(f"Average final log-likelihood: {avg_ll[-1]:.4f}\n")
        f.write(f"Std of final log-likelihood: {std_ll[-1]:.4f}\n")
        f.write(f"Best final log-likelihood: {best_ll:.4f}\n")
        f.write(f"Worst final log-likelihood: {min(t[0] for t in final_params):.4f}\n")
        f.write(f"Best run mixing weights pi: {np.round(best_pi, 4).tolist()}\n")

    print("Saved plot to plots/q1_em_loglikelihood.png")


if __name__ == "__main__":
    main()
