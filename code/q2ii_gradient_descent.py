"""
Assignment 2, Question 2(ii)
------------------------------------------------------------------
Batch Gradient Descent for least squares:
    L(w) = ||Xw - y||^2
    grad L(w) = 2 X^T (Xw - y)

Step size: L(w) = ||Xw-y||^2 is a convex quadratic whose Hessian is
H = 2 X^T X. The gradient is Lipschitz with constant Lip = 2*lambda_max(H/2)
= 2*lambda_max(X^T X). For a quadratic, fixed-step gradient descent
converges for any 0 < eta < 2/Lip_grad, and the fastest FIXED step is the
classic optimal choice
    eta* = 2 / (lambda_min(X^T X) + lambda_max(X^T X))
(derived from the 1-D optimal-step analysis applied to each eigen-
direction of the quadratic). We find lambda_min and lambda_max via the
power method (run on X^T X and on (lambda_max I - X^T X) respectively),
implemented from scratch with only basic linear algebra operations.

We plot || w_t - w_ML ||_2 as a function of iteration t.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from q2_common import load_xy, analytical_ols, TRAIN_PATH

N_ITERS = 500
RNG_SEED = 0


def power_method_largest_eig(A, n_iters=500, rng=None):
    """Largest eigenvalue of symmetric PSD matrix A via the power method."""
    rng = rng or np.random.default_rng(0)
    n = A.shape[0]
    v = rng.normal(size=n)
    v /= np.linalg.norm(v)
    eig = None
    for _ in range(n_iters):
        v_new = A @ v
        eig = np.linalg.norm(v_new)
        v = v_new / eig
    return eig


def power_method_smallest_eig(A, lambda_max, n_iters=500, rng=None):
    """Smallest eigenvalue of symmetric PSD A via power method on the
    shifted matrix (lambda_max*I - A), whose largest eigenvalue is
    lambda_max - lambda_min(A)."""
    n = A.shape[0]
    shifted = lambda_max * np.eye(n) - A
    top = power_method_largest_eig(shifted, n_iters=n_iters, rng=rng)
    return lambda_max - top


def gradient_descent(X, y, w_ml, n_iters, eta):
    n, d = X.shape
    w = np.zeros(d)
    dist_trace = np.zeros(n_iters + 1)
    dist_trace[0] = np.linalg.norm(w - w_ml)
    for t in range(1, n_iters + 1):
        grad = 2 * X.T @ (X @ w - y)
        w = w - eta * grad
        dist_trace[t] = np.linalg.norm(w - w_ml)
    return w, dist_trace


def main():
    X, y = load_xy(TRAIN_PATH)
    w_ml = analytical_ols(X, y)

    A = X.T @ X
    rng = np.random.default_rng(RNG_SEED)
    lambda_max = power_method_largest_eig(A, n_iters=500, rng=rng)
    lambda_min = power_method_smallest_eig(A, lambda_max, n_iters=500, rng=rng)
    # Loss = ||Xw-y||^2 has Hessian H = 2 X^T X, so eigenvalues of H are
    # 2*lambda_min(A), 2*lambda_max(A). The classic optimal fixed step for
    # quadratic minimization is eta* = 2 / (lambda_min(H)+lambda_max(H))
    #                                = 1 / (lambda_min(A)+lambda_max(A)).
    eta = 1.0 / (lambda_min + lambda_max)
    print(f"lambda_max(X^T X) = {lambda_max:.4f}")
    print(f"lambda_min(X^T X) = {lambda_min:.4f}")
    print(f"Condition number kappa = {lambda_max/lambda_min:.2f}")
    print(f"Step size eta* = 1/(lambda_min+lambda_max) = {eta:.8f}")

    w_gd, dist_trace = gradient_descent(X, y, w_ml, N_ITERS, eta)

    print(f"Final ||w_T - w_ML||_2 after {N_ITERS} iters = {dist_trace[-1]:.6f}")
    print(f"||w_ML||_2 = {np.linalg.norm(w_ml):.6f}")

    iters = np.arange(0, N_ITERS + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(iters, dist_trace, color="#2ca02c", linewidth=2)
    plt.xlabel("Iteration t")
    plt.ylabel(r"$\|w_t - w_{ML}\|_2$")
    plt.title(f"Gradient Descent convergence to w_ML  (η* = 1/(λmin+λmax) = {eta:.2e})")
    plt.yscale("log")
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("/home/claude/assignment2/plots/q2_gd_convergence.png", dpi=150)
    plt.close()

    np.savetxt("/home/claude/assignment2/plots/q2_gd_dist_trace.csv",
               np.column_stack([iters, dist_trace]), delimiter=",",
               header="iteration,dist_to_wml", comments="")

    with open("/home/claude/assignment2/plots/q2_gd_summary.txt", "w") as f:
        f.write(f"lambda_max(X^TX) = {lambda_max:.6f}\n")
        f.write(f"lambda_min(X^TX) = {lambda_min:.6f}\n")
        f.write(f"condition number = {lambda_max/lambda_min:.4f}\n")
        f.write(f"step size eta = {eta:.10f}\n")
        f.write(f"N_ITERS = {N_ITERS}\n")
        f.write(f"Final dist ||w_T - w_ML||_2 = {dist_trace[-1]:.8f}\n")
        f.write(f"Initial dist ||w_0 - w_ML||_2 = {dist_trace[0]:.8f}\n")

    print("Saved plot to plots/q2_gd_convergence.png")


if __name__ == "__main__":
    main()
