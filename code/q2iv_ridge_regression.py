"""
Assignment 2, Question 2(iv)
------------------------------------------------------------------
Ridge regression via gradient descent, with K-fold cross-validation to
choose the regularization strength lambda.

Loss:  L(w) = ||Xw - y||^2 + lambda * ||w||^2
Grad:  grad L(w) = 2 X^T (Xw - y) + 2 * lambda * w
Hessian: H = 2 X^T X + 2 lambda I  (always positive definite for lambda>0,
         so ridge regression is better conditioned / easier for GD than
         plain OLS).

Optimal fixed GD step size (same quadratic-minimization argument as
Q2(ii)):
    eta*(lambda) = 2 / (lambda_min(H) + lambda_max(H))
                 = 1 / (lambda_min(X^T X) + lambda_max(X^T X) + 2*lambda)

Procedure:
  1. 5-fold cross-validation on the training set for a grid of lambda
     values (log-spaced). For each (lambda, fold) we run ridge GD to
     convergence on the 4 training folds and evaluate MSE on the held
     out fold.
  2. Plot average CV validation error vs. lambda.
  3. Refit w_R on the FULL training set using the best lambda.
  4. Compare test-set error of w_R vs. w_ML (from Q2i) on A2Q2Data_test.csv.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from q2_common import load_xy, analytical_ols, mse, TRAIN_PATH, TEST_PATH

N_FOLDS = 5
LAMBDA_GRID = np.logspace(-3, 3, 25)  # 0.001 ... 1000
N_GD_ITERS = 3000
RNG_SEED = 2


def eig_extremes(A):
    """Smallest & largest eigenvalue of symmetric PSD matrix A (d=100 here,
    so a direct, exact eigen-decomposition is cheap and numerically exact --
    this is a linear-algebra primitive, not the learning algorithm itself)."""
    eigs = np.linalg.eigvalsh(A)
    return eigs[0], eigs[-1]


def ridge_gd(X, y, lam, n_iters, lambda_min_A, lambda_max_A, w_init=None):
    n, d = X.shape
    w = np.zeros(d) if w_init is None else w_init.copy()
    eta = 1.0 / (lambda_min_A + lambda_max_A + 2 * lam)
    for _ in range(n_iters):
        grad = 2 * X.T @ (X @ w - y) + 2 * lam * w
        w = w - eta * grad
    return w


def cross_validate(X, y, lambda_grid, n_folds, n_iters, rng):
    n = X.shape[0]
    perm = rng.permutation(n)
    fold_size = n // n_folds
    folds = [perm[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    # last fold absorbs remainder
    folds[-1] = np.concatenate([folds[-1], perm[n_folds * fold_size:]])

    cv_errors = np.zeros(len(lambda_grid))
    cv_errors_std = np.zeros(len(lambda_grid))

    for li, lam in enumerate(lambda_grid):
        fold_errs = []
        for f in range(n_folds):
            val_idx = folds[f]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            A_tr = X_tr.T @ X_tr
            lmin, lmax = eig_extremes(A_tr)
            w = ridge_gd(X_tr, y_tr, lam, n_iters, lmin, lmax)
            fold_errs.append(mse(X_val, y_val, w))
        cv_errors[li] = np.mean(fold_errs)
        cv_errors_std[li] = np.std(fold_errs)
        print(f"  lambda={lam:10.4f}  CV-MSE={cv_errors[li]:.6f} (+/- {cv_errors_std[li]:.6f})")

    return cv_errors, cv_errors_std


def main():
    X, y = load_xy(TRAIN_PATH)
    X_test, y_test = load_xy(TEST_PATH)
    w_ml = np.load("/home/claude/assignment2/plots/q2_w_ml.npy")

    rng = np.random.default_rng(RNG_SEED)
    print("Running 5-fold cross-validation over lambda grid...")
    cv_errors, cv_errors_std = cross_validate(X, y, LAMBDA_GRID, N_FOLDS, N_GD_ITERS, rng)

    best_idx = np.argmin(cv_errors)
    best_lambda = LAMBDA_GRID[best_idx]
    print(f"\nBest lambda (min CV error) = {best_lambda:.5f}, CV-MSE = {cv_errors[best_idx]:.6f}")

    # Plot validation error vs lambda
    plt.figure(figsize=(7, 5))
    plt.errorbar(LAMBDA_GRID, cv_errors, yerr=cv_errors_std, marker="o", markersize=4,
                 color="#e377c2", ecolor="lightgray", capsize=2, linewidth=1.5)
    plt.axvline(best_lambda, color="red", linestyle="--", alpha=0.7,
                label=f"best λ = {best_lambda:.4f}")
    plt.xscale("log")
    plt.xlabel("λ (regularization strength, log scale)")
    plt.ylabel("5-fold cross-validation MSE")
    plt.title("Ridge Regression: validation error vs. λ")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("/home/claude/assignment2/plots/q2_ridge_cv_error.png", dpi=150)
    plt.close()

    # Refit on FULL training data with best lambda
    A_full = X.T @ X
    lmin_full, lmax_full = eig_extremes(A_full)
    w_R = ridge_gd(X, y, best_lambda, n_iters=2000, lambda_min_A=lmin_full, lambda_max_A=lmax_full)

    train_mse_R = mse(X, y, w_R)
    test_mse_R = mse(X_test, y_test, w_R)
    train_mse_ML = mse(X, y, w_ml)
    test_mse_ML = mse(X_test, y_test, w_ml)

    print(f"\n--- Final comparison on A2Q2Data_test.csv ---")
    print(f"w_ML : train MSE = {train_mse_ML:.6f}, test MSE = {test_mse_ML:.6f}, ||w||={np.linalg.norm(w_ml):.4f}")
    print(f"w_R  : train MSE = {train_mse_R:.6f}, test MSE = {test_mse_R:.6f}, ||w||={np.linalg.norm(w_R):.4f}  (lambda={best_lambda:.5f})")

    np.save("/home/claude/assignment2/plots/q2_w_ridge.npy", w_R)
    np.savetxt("/home/claude/assignment2/plots/q2_ridge_cv_curve.csv",
               np.column_stack([LAMBDA_GRID, cv_errors, cv_errors_std]), delimiter=",",
               header="lambda,cv_mse,cv_mse_std", comments="")

    with open("/home/claude/assignment2/plots/q2_ridge_summary.txt", "w") as f:
        f.write(f"Lambda grid: {LAMBDA_GRID.tolist()}\n")
        f.write(f"CV errors: {cv_errors.tolist()}\n")
        f.write(f"Best lambda: {best_lambda}\n")
        f.write(f"Best lambda CV-MSE: {cv_errors[best_idx]}\n\n")
        f.write(f"w_ML train MSE: {train_mse_ML:.6f}\n")
        f.write(f"w_ML test MSE: {test_mse_ML:.6f}\n")
        f.write(f"||w_ML||_2: {np.linalg.norm(w_ml):.6f}\n\n")
        f.write(f"w_R train MSE: {train_mse_R:.6f}\n")
        f.write(f"w_R test MSE: {test_mse_R:.6f}\n")
        f.write(f"||w_R||_2: {np.linalg.norm(w_R):.6f}\n")

    print("Saved plot to plots/q2_ridge_cv_error.png")


if __name__ == "__main__":
    main()
