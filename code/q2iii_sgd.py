"""
Assignment 2, Question 2(iii)
------------------------------------------------------------------
Mini-batch Stochastic Gradient Descent (batch size = 100) for least
squares, implemented from scratch.

We minimize the mean-squared loss
    L(w) = (1/N) * ||Xw - y||^2 = (1/N) sum_n (x_n^T w - y_n)^2
(the minimizer of L is identical to the minimizer of the un-normalized
sum-of-squares loss used in Q2(ii), i.e. the same w_ML -- normalizing by
N only rescales the loss, not its minimizer).

At each step t we draw a random mini-batch B_t of size m=100 (without
replacement, re-shuffling every epoch) and take the step
    w_{t+1} = w_t - eta_t * (2/m) * X_{B_t}^T (X_{B_t} w_t - y_{B_t})
which is an unbiased stochastic estimate of the full gradient of L.

Step size: a DECAYING step size eta_t = eta0 / (1 + t/t0) is used
(the mini-batch gradient is a *noisy* estimate of the true gradient, so
a constant step size only gets w_t into a neighbourhood of w_ML and
keeps oscillating there -- eta_t -> 0 is required for w_t -> w_ML
exactly, consistent with the standard Robbins-Monro conditions
sum eta_t = inf, sum eta_t^2 < inf).

We plot || w_t - w_ML ||_2 as a function of iteration (mini-batch update) t.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from q2_common import load_xy, analytical_ols, TRAIN_PATH
from q2ii_gradient_descent import power_method_largest_eig, power_method_smallest_eig

BATCH_SIZE = 100
N_EPOCHS = 60
RNG_SEED = 1
T0 = 3000  # decay horizon for step size schedule


def sgd(X, y, w_ml, batch_size, n_epochs, eta0, t0, rng):
    n, d = X.shape
    w = np.zeros(d)
    n_batches_per_epoch = n // batch_size
    dist_trace = [np.linalg.norm(w - w_ml)]
    t = 0
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for b in range(n_batches_per_epoch):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            Xb, yb = X[idx], y[idx]
            grad = (2.0 / batch_size) * Xb.T @ (Xb @ w - yb)
            eta_t = eta0 / (1 + t / t0)
            w = w - eta_t * grad
            t += 1
            dist_trace.append(np.linalg.norm(w - w_ml))
    return w, np.array(dist_trace)


def main():
    X, y = load_xy(TRAIN_PATH)
    w_ml = analytical_ols(X, y)
    N, D = X.shape

    A = X.T @ X
    rng_eig = np.random.default_rng(0)
    lambda_max = power_method_largest_eig(A, n_iters=500, rng=rng_eig)
    lambda_min = power_method_smallest_eig(A, lambda_max, n_iters=500, rng=rng_eig)
    # Optimal fixed step for *mean*-loss full GD -> used as eta0 for SGD schedule
    eta0 = N / (lambda_min + lambda_max)
    print(f"eta0 (initial SGD step size) = {eta0:.6f}")

    rng = np.random.default_rng(RNG_SEED)
    w_sgd, dist_trace = sgd(X, y, w_ml, BATCH_SIZE, N_EPOCHS, eta0, T0, rng)

    print(f"Total mini-batch updates: {len(dist_trace)-1}")
    print(f"Final ||w_t - w_ML||_2 = {dist_trace[-1]:.6f}")

    iters = np.arange(len(dist_trace))
    plt.figure(figsize=(7, 5))
    plt.plot(iters, dist_trace, color="#9467bd", linewidth=1)
    plt.xlabel("Iteration t  (mini-batch updates, batch size = 100)")
    plt.ylabel(r"$\|w_t - w_{ML}\|_2$")
    plt.title(f"SGD (batch=100) convergence to w_ML over {N_EPOCHS} epochs")
    plt.yscale("log")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("/home/claude/assignment2/plots/q2_sgd_convergence.png", dpi=150)
    plt.close()

    np.savetxt("/home/claude/assignment2/plots/q2_sgd_dist_trace.csv",
               np.column_stack([iters, dist_trace]), delimiter=",",
               header="iteration,dist_to_wml", comments="")

    with open("/home/claude/assignment2/plots/q2_sgd_summary.txt", "w") as f:
        f.write(f"batch_size={BATCH_SIZE}, n_epochs={N_EPOCHS}, eta0={eta0:.6f}, t0={T0}\n")
        f.write(f"Total updates: {len(dist_trace)-1}\n")
        f.write(f"Final dist ||w_T - w_ML||_2 = {dist_trace[-1]:.8f}\n")
        f.write(f"Initial dist ||w_0 - w_ML||_2 = {dist_trace[0]:.8f}\n")

    print("Saved plot to plots/q2_sgd_convergence.png")


if __name__ == "__main__":
    main()
