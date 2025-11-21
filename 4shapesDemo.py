import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_add_pool
import networkx as nx
import numpy as np
import random, math, os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# -----------------------------
# Repro
# -----------------------------
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
os.makedirs(".", exist_ok=True)

# ==========================================================
# 1. Graph utilities
# ==========================================================
def graph_to_data(G, label):
    """Convert NX graph to PyG Data with degree features"""
    if not all(isinstance(n, int) for n in G.nodes()):
        G = nx.convert_node_labels_to_integers(G)
    degs = np.array([d for _, d in G.degree()]).reshape(-1, 1)
    x = torch.tensor(degs, dtype=torch.float32)
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    if edge_index.numel() > 0:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label]))

# Motif generators
def make_star():     return nx.star_graph(random.randint(3, 6))
def make_grid():     return nx.grid_2d_graph(random.randint(2,3), random.randint(2,3))
def make_lollipop(): return nx.lollipop_graph(random.randint(3,4), random.randint(1,3))
def make_tree():     return nx.balanced_tree(2, random.randint(2,3))

# Attach motifs to BA backbone
def make_class_graph(cls, n_motifs=3):
    if cls == 0: motif_fn = make_star
    elif cls == 1: motif_fn = make_grid
    elif cls == 2: motif_fn = make_lollipop
    else:          motif_fn = make_tree

    G = nx.barabasi_albert_graph(random.randint(4, 20), 3)
    offset = len(G)
    for _ in range(n_motifs):
        mot = motif_fn()
        G = nx.disjoint_union(G, mot)
        motif_nodes = [i for i in range(offset, offset + len(mot))]
        ba_nodes    = [i for i in range(offset)]
        G.add_edge(random.choice(ba_nodes), random.choice(motif_nodes))
        offset = len(G)
    return graph_to_data(G, cls)

def make_class_graph_nx(cls, n_motifs=3):
    """
    Build a BA + multiple motifs graph for class `cls`, and return:
    - G: networkx graph
    - node_types: dict {node: "ba" or "motif"}
    """
    if cls == 0:
        motif_fn = make_star
    elif cls == 1:
        motif_fn = make_grid
    elif cls == 2:
        motif_fn = make_lollipop
    else:
        motif_fn = make_tree

    # BA backbone
    G_ba = nx.barabasi_albert_graph(random.randint(10, 30), 3)
    G = G_ba.copy()
    node_types = {n: "ba" for n in G.nodes()}

    # Keep track of BA nodes (they remain the initial nodes)
    ba_nodes = list(G_ba.nodes())
    next_id = max(G.nodes()) + 1 if len(G) > 0 else 0

    # Attach n_motifs copies of the motif
    for _ in range(n_motifs):
        mot = motif_fn()
        # Relabel motif nodes to fresh integer IDs starting from next_id
        mot = nx.convert_node_labels_to_integers(mot, first_label=next_id)
        G.add_nodes_from(mot.nodes())
        G.add_edges_from(mot.edges())

        for n in mot.nodes():
            node_types[n] = "motif"

        motif_nodes = list(mot.nodes())
        # Connect one BA node to one motif node
        G.add_edge(random.choice(ba_nodes), random.choice(motif_nodes))
        next_id = max(G.nodes()) + 1

    return G, node_types



# Dataset: BA + multiple motifs per class
dataset = []
for cls in range(4):
    dataset += [make_class_graph(cls, n_motifs=random.randint(2, 8)) for _ in range(1000)]
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ==========================================================
# 2. GCN classifier
# ==========================================================
class ToyGCN(nn.Module):
    def __init__(self, in_dim=1, hidden=64, num_classes=4):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin   = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        g = global_add_pool(x, batch)
        return self.lin(g)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = ToyGCN().to(device)
opt    = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-3)
crit   = nn.CrossEntropyLoss()

# Train
for epoch in range(60):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        logits = model(batch)
        loss = crit(logits, batch.y)
        loss.backward()
        opt.step()
        total += loss.item()
    if epoch % 10 == 0 or epoch == 59:
        print(f"Epoch {epoch:03d} | Loss {total/len(loader):.3f}")

# Confusion matrix
model.eval()
y_true, y_pred = [], []
for data in dataset:
    data = data.to(device)
    pred = model(data).argmax(dim=-1).item()
    y_true.append(data.y.item())
    y_pred.append(pred)
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

# ==========================================================
# 3. Embeddings + metrics helpers (ANGULAR)
# ==========================================================
@torch.no_grad()
def get_embeddings(model, dataset, normalize=False):
    """
    Graph-level embeddings (sum pooled), plus labels.
    When normalize=True, embeddings are L2-normalized and we use angular distances.
    """
    model.eval()
    vecs, labels = [], []
    for data in dataset:
        data = data.to(device)
        x = F.relu(model.conv1(data.x, data.edge_index))
        x = F.relu(model.conv2(x, data.edge_index))
        g = global_add_pool(x, torch.zeros(data.num_nodes, dtype=torch.long, device=device))
        vecs.append(g.cpu().squeeze())
        labels.append(int(data.y.item()))
    Z = torch.stack(vecs, dim=0)
    if normalize:
        Z = F.normalize(Z, p=2, dim=-1)
    return Z.cpu().numpy(), np.array(labels)

def pairwise_angle2(A, B):
    """
    Squared angular distances between rows of A (n,d) and B (m,d).
    Assumes A and B are L2-normalized. Returns theta^2 in radians^2.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim == 1:
        A = A[None, :]
    if B.ndim == 1:
        B = B[None, :]
    n, dA = A.shape
    m, dB = B.shape
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)
    assert dA == dB, f"Dim mismatch: {dA} vs {dB}"
    # cosine similarities (since A,B are unit vectors)
    S = A @ B.T
    S = np.clip(S, -1.0, 1.0)
    Theta = np.arccos(S)   # radians
    return Theta**2

def greedy_cover_sets(dist2, r2, K_max=None):
    """Greedy selection of motifs to maximize coverage; returns alpha sequence and sets."""
    N, M = dist2.shape
    if N == 0 or M == 0:
        return [], []
    K_max = M if K_max is None else min(K_max, M)
    covered = np.zeros(N, dtype=bool)
    sets    = []
    alpha   = []
    remaining = set(range(M))
    for _ in range(K_max):
        best, best_gain, best_mask = None, -1, None
        for j in remaining:
            mask = (dist2[:, j] <= r2) & (~covered)
            gain = int(mask.sum())
            if gain > best_gain:
                best, best_gain, best_mask = j, gain, mask
        if best is None:
            break
        remaining.remove(best)
        covered |= best_mask
        sets.append(dist2[:, best] <= r2)
        alpha.append(float(covered.mean()))
        if covered.all():
            break
    return alpha, sets

def compute_overlap(sets, n_total):
    """Multi-coverage overlap: (sum multiplicity - union)/union."""
    if not sets:
        return 0.0
    M = np.stack(sets, axis=1)            # (N, K)
    sum_mult = float(M.sum(axis=1).sum())
    union    = float((M.any(axis=1)).sum())
    if union <= 0:
        return 0.0
    return (sum_mult - union) / union

def hoeffding_halfwidth(n, delta=0.05):
    return math.sqrt((1.0/(2.0*max(1, n))) * math.log(2.0/delta))

def coverage_gga_overlap(pos_embs, motif_embs, r_theta):
    """
    Coverage, GGA, Overlap at angular radius r_theta (radians).
    Distances are angular: dist2 = theta^2 and we threshold at r_theta^2.
    """
    r2 = r_theta * r_theta
    D2 = pairwise_angle2(pos_embs, motif_embs)
    if D2.size == 0:
        return 0.0, 0.0, 0.0, []
    cov = float((np.min(D2, axis=1) <= r2).mean())
    alpha, sets = greedy_cover_sets(D2, r2)
    gga = float(np.mean(alpha)) if alpha else 0.0
    ovl = compute_overlap(sets, pos_embs.shape[0]) if sets else 0.0
    return cov, gga, ovl, alpha

# ==========================================================
# 4. Explanations
# ==========================================================
def generate_good_motifs(n=8):
    return {
        0: [make_star()     for _ in range(n)],
        1: [make_grid()     for _ in range(n)],
        2: [make_lollipop() for _ in range(n)],
        3: [make_tree()     for _ in range(n)],
    }

def generate_random_motifs(n=8):
    return {
        c: [nx.barabasi_albert_graph(random.randint(10, 15), 3) for _ in range(n)]
        for c in range(4)
    }

good_motifs   = generate_good_motifs(5)
random_motifs = generate_random_motifs(5)

@torch.no_grad()
def motif_scores(model, motifs, c):
    scores = []
    for G in motifs:
        data = graph_to_data(G, c).to(device)
        p    = F.softmax(model(data), dim=-1).cpu().numpy().squeeze()
        scores.append(float(p[c]))
    return np.mean(scores), np.std(scores)

def motifs_to_embeddings(motifs, c, normalize=False):
    """Compute embeddings for motifs treated as class c."""
    embs = []
    for G in motifs:
        data = graph_to_data(G, c)
        vec, _ = get_embeddings(model, [data], normalize=normalize)
        embs.append(vec.squeeze(0))
    return np.stack(embs)

# ==========================================================
# 5. Metrics (ANGULAR embeddings, mapped radius)
# ==========================================================
# Use L2-normalized embeddings (unit vectors) for angular formulation
embs_ang, labels = get_embeddings(model, dataset, normalize=True)

results = []
alpha_curves = {}
#print("\n=== Metrics with ANGULAR distances (normalized embeddings) ===")
for set_name, mset in [("Good", good_motifs), ("Random", random_motifs)]:
    print(f"\n--- {set_name} motifs ---")
    for c in range(4):
        pos   = embs_ang[labels == c]
        m_emb = motifs_to_embeddings(mset[c], c, normalize=True)

        # Class-wise Euclidean radius r_e = 1/(2||w_c||), then map to angular radius r_theta
        W  = model.lin.weight.detach().cpu().numpy()
        Lc = np.linalg.norm(W[c], 2) + 1e-12
        r_e = 1.0 / (2.0 * Lc)          # Euclidean optimal radius from theorem
        arg = 1.0 - 0.5 * (r_e ** 2)    # cos(theta) = 1 - r_e^2 / 2 on unit sphere
        arg = np.clip(arg, -1.0, 1.0)
        r_theta = math.acos(arg)        # angular radius (radians), matching r_e

        cov, gga, ovl, alpha = coverage_gga_overlap(pos, m_emb, r_theta)
        hw = hoeffding_halfwidth(len(pos))
        mu, std = motif_scores(model, mset[c], c)

        print(
            f"Class {c} | {set_name} | r_e={r_e:.3f}, r_theta={r_theta:.3f} "
            f"| Score={mu:.3f}±{std:.3f} | Coverage={cov:.3f} ±{hw:.3f} "
            f"| GGA={gga:.3f} | Overlap={ovl:.3f}"
        )

        alpha_curves[(c, set_name)] = alpha
        results.append({
            "Class": c,
            "Set": set_name,
            "ScoreMean": mu,
            "ScoreStd": std,
            "Coverage": cov,
            "CovHW": hw,
            "GGA": gga,
            "Overlap": ovl,
            "r_e": r_e,
            "r_theta": r_theta,
        })

results_df = pd.DataFrame(results)

# ==========================================================
# 6. Plots
# ==========================================================
palette = sns.color_palette("tab10", 4)
width   = 0.35
x       = np.arange(4)

# Fig A: Scores
plt.figure(figsize=(8, 6))
score_good = [results_df[(results_df.Class == c) & (results_df.Set == "Good")]["ScoreMean"].item()
              for c in range(4)]
score_rand = [results_df[(results_df.Class == c) & (results_df.Set == "Random")]["ScoreMean"].item()
              for c in range(4)]
std_good   = [results_df[(results_df.Class == c) & (results_df.Set == "Good")]["ScoreStd"].item()
              for c in range(4)]
std_rand   = [results_df[(results_df.Class == c) & (results_df.Set == "Random")]["ScoreStd"].item()
              for c in range(4)]
plt.bar(x - width/2, score_good, width, yerr=std_good, capsize=4,
        label="Good", color="#4c72b0")
plt.bar(x + width/2, score_rand, width, yerr=std_rand, capsize=4,
        label="Random", color="#dd8452")
plt.xticks(x, [f"Class {c}" for c in range(4)])
plt.ylabel("Class score")
plt.legend()
plt.tight_layout()
plt.savefig("4shapes_scores_hist.png", dpi=300)
plt.close()

# Fig B: Coverage
plt.figure(figsize=(8, 6))
cov_good = [results_df[(results_df.Class == c) & (results_df.Set == "Good")]["Coverage"].item()
            for c in range(4)]
cov_rand = [results_df[(results_df.Class == c) & (results_df.Set == "Random")]["Coverage"].item()
            for c in range(4)]
hw_good  = [results_df[(results_df.Class == c) & (results_df.Set == "Good")]["CovHW"].item()
            for c in range(4)]
hw_rand  = [results_df[(results_df.Class == c) & (results_df.Set == "Random")]["CovHW"].item()
            for c in range(4)]
plt.bar(x - width/2, cov_good, width, yerr=hw_good, capsize=4,
        label="Good", color="#4c72b0")
plt.bar(x + width/2, cov_rand, width, yerr=hw_rand, capsize=4,
        label="Random", color="#dd8452")
for i, val in enumerate(cov_rand):
    if val == 0:
        plt.plot(x[i] + width/2, 0.02, marker="_", color="black", markersize=12,
                 label="Zero coverage" if i == 0 else "")
plt.xticks(x, [f"Class {c}" for c in range(4)])
plt.ylabel("Coverage")
plt.legend()
plt.tight_layout()
plt.savefig("4shapes_coverage_hist.png", dpi=300)
plt.close()

# Fig C: GGA curves
plt.figure(figsize=(8, 6))
K = max(len(v) for v in alpha_curves.values())
for c in range(4):
    a_g = alpha_curves[(c, "Good")]
    a_r = alpha_curves[(c, "Random")]
    a_g = a_g + [a_g[-1] if a_g else 0] * (K - len(a_g))
    a_r = a_r + [a_r[-1] if a_r else 0] * (K - len(a_r))
    plt.plot(range(1, K+1), a_g, color=palette[c], lw=2)
    plt.plot(range(1, K+1), a_r, color=palette[c], lw=2, ls="--")
plt.xlabel("# motifs")
plt.ylabel("Coverage")
plt.title("GGA curves")
plt.tight_layout()
plt.savefig("4shapes_gga_allclasses.png", dpi=300)
plt.close()

print("Saved plots: scores, coverage, gga curves, confusion_matrix.png")



