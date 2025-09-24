import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
import random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_add_pool

# import metric functions from metrics.py
from metrics import (
    get_embeddings, normalize_rows,
    angular_distance_matrix, coverage_gga_overlap,
    compute_lipschitz_linear_head, compute_r_star_ang,
    hoeffding_ci_coverage, hoeffding_ci_gga
)


# ==========================================================
# 1. Graph utilities (motif-based 4shapes dataset)
# ==========================================================
def graph_to_data(G, label):
    if not all(isinstance(n, int) for n in G.nodes()):
        G = nx.convert_node_labels_to_integers(G)
    degs = np.array([d for _, d in G.degree()]).reshape(-1, 1)
    x = torch.tensor(degs, dtype=torch.float32)
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    if edge_index.numel() > 0:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label]))


def make_star(): return nx.star_graph(random.randint(3, 6))
def make_grid(): return nx.grid_2d_graph(random.randint(2, 3), random.randint(2, 3))
def make_lollipop(): return nx.lollipop_graph(random.randint(3, 4), random.randint(1, 3))
def make_tree(): return nx.balanced_tree(2, random.randint(2, 3))


def make_class_graph(cls, n_motifs=3):
    if cls == 0: motif_fn = make_star
    elif cls == 1: motif_fn = make_grid
    elif cls == 2: motif_fn = make_lollipop
    else: motif_fn = make_tree

    G = nx.barabasi_albert_graph(random.randint(4, 10), 3)
    offset = len(G)
    for _ in range(n_motifs):
        mot = motif_fn()
        G = nx.disjoint_union(G, mot)
        motif_nodes = [i for i in range(offset, offset + len(mot))]
        ba_nodes = [i for i in range(offset)]
        G.add_edge(random.choice(ba_nodes), random.choice(motif_nodes))
        offset = len(G)
    return graph_to_data(G, cls)


# ==========================================================
# 2. Toy GCN
# ==========================================================
class ToyGCN(nn.Module):
    def __init__(self, in_dim=1, hidden=64, num_classes=4):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        g = global_add_pool(x, batch)
        return self.lin(g)


# ==========================================================
# 3. Train GCN on 4shapes dataset
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = []
for cls in range(4):
    dataset += [make_class_graph(cls, n_motifs=random.randint(2, 6)) for _ in range(100)]
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ToyGCN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-3)
crit = nn.CrossEntropyLoss()

for epoch in range(5):  # short training just for demo
    model.train()
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        logits = model(batch)
        loss = crit(logits, batch.y)
        loss.backward()
        opt.step()


# ==========================================================
# 4. Compute embeddings and metrics
# ==========================================================
embs_raw, labels = get_embeddings(model, dataset, device)
embs_norm = normalize_rows(embs_raw)

# Lipschitz constant and angular radius
L = compute_lipschitz_linear_head(model)
r_star = compute_r_star_ang(L)

# Pick one class (e.g. class 0) and one motif set
pos = embs_norm[labels == 0]
motifs = [make_star() for _ in range(5)]
motif_embs, _ = get_embeddings(model, [graph_to_data(G, 0) for G in motifs], device)

D = angular_distance_matrix(pos, motif_embs)
cov, gga, ovl = coverage_gga_overlap(D, r_star)

# Hoeffding confidence intervals
n = pos.shape[0]
K = len(motifs)
cov_ci = hoeffding_ci_coverage(cov, n, delta=0.05)
gga_ci = hoeffding_ci_gga(gga, n, K, delta=0.05)

print("=== Metrics on class 0 motifs (4shapes) ===")
print(f"Coverage: {cov:.3f} (95% CI: {cov_ci[0]:.3f} – {cov_ci[1]:.3f})")
print(f"GGA     : {gga:.3f} (95% CI: {gga_ci[0]:.3f} – {gga_ci[1]:.3f})")
print(f"Overlap : {ovl:.3f}")
