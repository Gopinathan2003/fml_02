"""
Assignment 2, Question 1(ii)
------------------------------------------------------------------
K-means clustering (Lloyd's algorithm) implemented from scratch,
run on A2Q1.csv with K = 4.

Objective (within-cluster sum of squares):
    J = sum_{n=1}^{N} sum_{k=1}^{K} r_nk * ||x_n - c_k||^2
where r_nk = 1 if x_n is assigned to cluster k (nearest centroid), else 0.

We run several random initializations (centroids = K distinct randomly
chosen data points) and plot the objective J as a function of iteration
for the run that achieves the lowest final objective (standard practice
to avoid reporting a poor local optimum), as well as save all runs for
reference.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 7
K = 4
N_ITERS = 30
N_RESTARTS = 20


def load_data(path):
    return np.loadtxt(path, delimiter=",")


def kmeans_objective(X, centroids, labels):
    diffs = X - centroids[labels]
    return np.sum(diffs ** 2)


def run_kmeans(X, K, n_iters, rng):
    N, D = X.shape
    init_idx = rng.choice(N, size=K, replace=False)
    centroids = X[init_idx].copy()

    obj_trace = np.zeros(n_iters)
    labels = None
    for t in range(n_iters):
        # Assignment step: squared Euclidean distance to each centroid
        dists = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)  # (N,K)
        labels = np.argmin(dists, axis=1)

        obj_trace[t] = kmeans_objective(X, centroids, labels)

        # Update step
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            members = X[labels == k]
            if len(members) > 0:
                new_centroids[k] = members.mean(axis=0)
            else:
                # re-seed empty cluster to a random data point
                new_centroids[k] = X[rng.integers(0, N)]
        centroids = new_centroids

    return obj_trace, centroids, labels


def main():
    X = load_data("/home/claude/assignment2/data/A2Q1.csv")
    N, D = X.shape
    print(f"Loaded data: N={N}, D={D}")

    master_rng = np.random.default_rng(RNG_SEED)
    all_traces = []
    best_trace, best_final_obj, best_labels = None, np.inf, None

    for r in range(N_RESTARTS):
        rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1))
        trace, centroids, labels = run_kmeans(X, K, N_ITERS, rng)
        all_traces.append(trace)
        if trace[-1] < best_final_obj:
            best_final_obj = trace[-1]
            best_trace = trace
            best_labels = labels
        print(f"  restart {r+1}/{N_RESTARTS}: final objective = {trace[-1]:.2f}")

    all_traces = np.array(all_traces)
    cluster_sizes = np.bincount(best_labels, minlength=K)
    print(f"\nBest final objective across {N_RESTARTS} restarts: {best_final_obj:.2f}")
    print(f"Cluster sizes of best run: {cluster_sizes}")

    # Plot best run
    iters = np.arange(1, N_ITERS + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(iters, best_trace, color="#d62728", linewidth=2, marker="o", markersize=3,
              label="K-means objective (best of 20 restarts)")
    plt.xlabel("Iteration")
    plt.ylabel("K-means objective  J = Σ ||x_n − c_{k(n)}||²")
    plt.title("K-means (K=4) objective vs. iteration on A2Q1.csv")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/claude/assignment2/plots/q1_kmeans_objective.png", dpi=150)
    plt.close()

    # Also plot all restarts together (to show sensitivity to initialization)
    plt.figure(figsize=(7, 5))
    for r in range(N_RESTARTS):
        plt.plot(iters, all_traces[r], alpha=0.35, linewidth=1, color="gray")
    plt.plot(iters, best_trace, color="#d62728", linewidth=2.5, label="Best restart")
    plt.xlabel("Iteration")
    plt.ylabel("K-means objective J")
    plt.title("K-means (K=4): all 20 restarts on A2Q1.csv")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/claude/assignment2/plots/q1_kmeans_all_restarts.png", dpi=150)
    plt.close()

    with open("/home/claude/assignment2/plots/q1_kmeans_summary.txt", "w") as f:
        f.write(f"N_RESTARTS={N_RESTARTS}, N_ITERS={N_ITERS}, K={K}\n")
        f.write(f"Best final objective: {best_final_obj:.4f}\n")
        f.write(f"Final objectives across restarts: {np.round(all_traces[:,-1],2).tolist()}\n")
        f.write(f"Cluster sizes (best run): {cluster_sizes.tolist()}\n")

    print("Saved plots to plots/q1_kmeans_objective.png and q1_kmeans_all_restarts.png")


if __name__ == "__main__":
    main()
