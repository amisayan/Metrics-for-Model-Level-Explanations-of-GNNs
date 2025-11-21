import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool


# ==========================================================
# 1. Embeddings
# ==========================================================
# Extract embeddings by forwarding through conv1/conv2 and pooling
@torch.no_grad()
def get_embeddings(model, dataset, device):
    model.eval()
    vecs, labels = [], []
    for data in dataset:
        data = data.to(device)
        x = F.relu(model.conv1(data.x, data.edge_index))
        x = F.relu(model.conv2(x, data.edge_index))
        g = global_add_pool(x, torch.zeros(data.num_nodes, dtype=torch.long, device=device))
        vecs.append(g.cpu().squeeze())
        labels.append(int(data.y.item()))
    return torch.stack(vecs, dim=0).numpy(), np.array(labels)


# Normalize rows of an embedding matrix
def normalize_rows(X, eps=1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


# ==========================================================
# 2. Distance + metrics
# ==========================================================
# Compute the angular distance matrix between embeddings
def angular_distance_matrix(A, B):
    cos = np.clip(
        A @ B.T / (np.linalg.norm(A, axis=1, keepdims=True) *
                   np.linalg.norm(B, axis=1, keepdims=True).T + 1e-12),
        -1.0, 1.0
    )
    return np.arccos(cos)


# Greedy covering to obtain prefix coverages
def greedy_cover_sets(D, r):
    N, M = D.shape
    covered = np.zeros(N, dtype=bool)
    sets, alpha = [], []
    remaining = set(range(M))
    while remaining:
        best, gain, best_mask = None, -1, None
        for j in remaining:
            mask = (D[:, j] <= r) & (~covered)
            g = int(mask.sum())
            if g > gain:
                best, gain, best_mask = j, g, mask
        if best is None:
            break
        remaining.remove(best)
        covered |= best_mask
        sets.append(D[:, best] <= r)
        alpha.append(covered.mean())
        if covered.all():
            break
    return alpha, sets


# Compute overlap as redundancy beyond union normalized by union size
def compute_overlap(sets):
    if not sets:
        return 0.0
    M = np.stack(sets, axis=1)
    sum_mult = float(M.sum(axis=1).sum())
    union = float((M.any(axis=1)).sum())
    return (sum_mult - union) / max(1.0, union)


# Compute Coverage, GGA, and Overlap
def coverage_gga_overlap(D, r):
    cov = (np.min(D, axis=1) <= r).mean() if D.size else 0.0
    alpha, sets = greedy_cover_sets(D, r)
    gga = np.mean(alpha) if alpha else 0.0
    ovl = compute_overlap(sets) if sets else 0.0
    return cov, gga, ovl


# ==========================================================
# 3. Lipschitz constant and angular radius
# ==========================================================
# Compute the Lipschitz constant of the linear layer as spectral norm of W
def compute_lipschitz_linear_head(model):
    W = model.lin.weight.detach().cpu().numpy()
    L = float(np.linalg.norm(W, 2)) + 1e-12
    return L


# Compute the angular radius r* = 2 * arcsin(1 / (4L))
def compute_r_star_ang(L):
    val = max(1.0 / (4.0 * max(L, 1e-12)), 0.0)
    val = min(val, 1.0)
    return float(2.0 * np.arcsin(val))


# ==========================================================
# 4. Hoeffding confidence intervals
# ==========================================================
# Coverage: |Cov_hat - Cov| ≤ sqrt( log(2/δ) / (2n) )
def hoeffding_ci_coverage(cov_hat, n, delta=0.05):
    eps = np.sqrt(np.log(2.0 / delta) / (2 * max(n, 1)))
    lower = max(0.0, cov_hat - eps)
    upper = min(1.0, cov_hat + eps)
    return lower, upper


# GGA: |GGA_hat - GGA| ≤ sqrt( log(2K/δ) / (2n) ), union bound over K steps
def hoeffding_ci_gga(gga_hat, n, K, delta=0.05):
    eps = np.sqrt(np.log((2.0 * max(K, 1)) / delta) / (2 * max(n, 1)))
    lower = max(0.0, gga_hat - eps)
    upper = min(1.0, gga_hat + eps)
    return lower, upper
