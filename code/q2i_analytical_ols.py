"""
Assignment 2, Question 2(i)
------------------------------------------------------------------
Analytical (closed-form) least-squares solution:
    w_ML = (X^T X)^{-1} X^T y
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from q2_common import load_xy, analytical_ols, mse, TRAIN_PATH, TEST_PATH


def main():
    X, y = load_xy(TRAIN_PATH)
    print(f"Train data: X={X.shape}, y={y.shape}")

    w_ml = analytical_ols(X, y)

    np.save("/home/claude/assignment2/plots/q2_w_ml.npy", w_ml)

    train_mse = mse(X, y, w_ml)
    print(f"||w_ML||_2 = {np.linalg.norm(w_ml):.4f}")
    print(f"Training MSE with w_ML = {train_mse:.6f}")

    X_test, y_test = load_xy(TEST_PATH)
    test_mse = mse(X_test, y_test, w_ml)
    print(f"Test MSE with w_ML = {test_mse:.6f}")

    with open("/home/claude/assignment2/plots/q2_wml_summary.txt", "w") as f:
        f.write(f"w_ML (first 10 components): {np.round(w_ml[:10], 4).tolist()}\n")
        f.write(f"||w_ML||_2 = {np.linalg.norm(w_ml):.6f}\n")
        f.write(f"Training MSE: {train_mse:.6f}\n")
        f.write(f"Test MSE: {test_mse:.6f}\n")


if __name__ == "__main__":
    main()
