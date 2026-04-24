"""
Plot the belief-enumeration DAG for a small Ramsey-like qubit POMDP.

K=3 epochs, J=2 delays, 1 shot per epoch.
- Count-tuple state: (N_1, M_1, N_2, M_2) where N_j = shots at delay j,
  M_j = count of y=1 outcomes.
- Likelihood: p(y=1 | x, τ) = ½ + ½ · v(x) · cos(x · τ), with x-dependent
  visibility v(x) = 1 - 0.3 x breaking the cos-symmetry.
- Prior: Gaussian-bumpy on x ∈ [0, 1] centered at x=0.35, σ=0.25.

Shows:
  (a) The DAG of reachable count-tuples across epochs. Edges color-coded
      by action (delay choice) and styled by outcome. Nodes with multiple
      parent paths (memoization collapse) are ringed in red.
  (b) Posterior densities at the root and at three representative tuples,
      illustrating how different count-tuples produce different posteriors.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

# -----------------------------------------------------------------------------
# Problem setup
# -----------------------------------------------------------------------------
K = 3                           # epochs
J = 2                           # number of delays
TAU = np.array([2.5,            # τ_1: slow oscillation
                6.0])           # τ_2: faster oscillation

# Grid for belief over x ∈ [0, 1]
Nx = 256
X = np.linspace(0.0, 1.0, Nx)
DX = X[1] - X[0]

# Prior: Gaussian-bump centered at 0.35, σ=0.25 (soft, but asymmetric)
PRIOR_MU = 0.35
PRIOR_SIGMA = 0.25
PRIOR = np.exp(-0.5 * ((X - PRIOR_MU) / PRIOR_SIGMA) ** 2)
PRIOR /= np.sum(PRIOR) * DX

def likelihood_p(x, tau):
    """p(y=1 | x, tau). Visibility v(x) = 1 - 0.3 x breaks cos-symmetry."""
    v = 1.0 - 0.3 * x
    return 0.5 + 0.5 * v * np.cos(x * tau)

def posterior(counts):
    """Posterior p(x | counts) on grid X."""
    log_p = np.log(PRIOR + 1e-300)
    for j, (N, M) in enumerate(counts):
        if N == 0:
            continue
        p1 = likelihood_p(X, TAU[j])
        p1 = np.clip(p1, 1e-12, 1 - 1e-12)
        log_p += M * np.log(p1) + (N - M) * np.log(1 - p1)
    log_p -= np.max(log_p)
    p = np.exp(log_p)
    p /= np.sum(p) * DX
    return p

def posterior_stats(p):
    """Posterior mean and standard deviation."""
    mean = np.sum(X * p) * DX
    var = np.sum((X - mean) ** 2 * p) * DX
    return mean, np.sqrt(var)

# -----------------------------------------------------------------------------
# BFS enumerate reachable count-tuples at each depth
# -----------------------------------------------------------------------------
empty = tuple((0, 0) for _ in range(J))
levels = [set([empty])]
edges = []  # (parent, child, action_j, outcome_y)

current = set([empty])
for k in range(K):
    nxt = set()
    for parent in current:
        for j in range(J):
            for y in (0, 1):
                N_j, M_j = parent[j]
                child = tuple(parent[:j] + ((N_j + 1, M_j + y),) + parent[j+1:])
                nxt.add(child)
                edges.append((parent, child, j, y))
    levels.append(nxt)
    current = nxt

print(f"Reachable count-tuples per epoch: {[len(L) for L in levels]}")
print(f"Total nodes: {sum(len(L) for L in levels)}")
print(f"Total edges: {len(edges)}")

# Count distinct parent paths per child
in_degree = defaultdict(int)
for parent, child, j, y in edges:
    in_degree[child] += 1
merged_nodes = {n for n, d in in_degree.items() if d > 1}
print(f"Merged (multi-parent) nodes: {len(merged_nodes)}")

# -----------------------------------------------------------------------------
# Layout: arrange nodes horizontally by epoch
# -----------------------------------------------------------------------------
def node_order_key(c):
    return (c[0][0], c[0][1], c[1][1])

pos = {}
for k, L in enumerate(levels):
    sorted_L = sorted(L, key=node_order_key)
    n = len(sorted_L)
    for i, node in enumerate(sorted_L):
        y_pos = (i - (n - 1) / 2) * 1.0
        pos[node] = (k * 3.5, y_pos)

# Compute posterior stats per node
stats = {}
for L in levels:
    for node in L:
        p = posterior(node)
        stats[node] = posterior_stats(p)

# -----------------------------------------------------------------------------
# Draw
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(36, 20))
# Global font scale
plt.rcParams.update({'font.size': 24})

# Left big panel: the DAG
ax_tree = plt.subplot2grid((4, 5), (0, 0), colspan=4, rowspan=4)

ACTION_COLOR = {0: '#2E86AB', 1: '#C73E1D'}
OUTCOME_STYLE = {0: (0, (3, 2)),   # dashed for y=0 (miss)
                 1: '-'}           # solid for y=1 (detect)

# Edges
for parent, child, j, y in edges:
    x1, yy1 = pos[parent]
    x2, yy2 = pos[child]
    ax_tree.plot([x1, x2], [yy1, yy2],
                 color=ACTION_COLOR[j],
                 linestyle=OUTCOME_STYLE[y],
                 alpha=0.55,
                 linewidth=3.0,
                 zorder=1)

# Nodes
import matplotlib.cm as cm
min_sigma = min(s for _, s in stats.values())
max_sigma = max(s for _, s in stats.values())
sigma_range = max_sigma - min_sigma if max_sigma > min_sigma else 1.0
min_mean = min(m for m, _ in stats.values())
max_mean = max(m for m, _ in stats.values())

for node, (x, yy) in pos.items():
    mean, sigma = stats[node]
    # Size: larger for sharper posterior
    norm_size = 1.0 - (sigma - min_sigma) / sigma_range
    size = 500 + 1800 * norm_size
    # Color: posterior mean, normalized to its range
    color_val = (mean - min_mean) / (max_mean - min_mean + 1e-12)
    color = cm.plasma(color_val)
    # Edge highlight if this node has multiple parent paths
    merge = node in merged_nodes
    edgecolor = 'black' if not merge else 'red'
    linewidth = 1.8 if not merge else 5.0
    ax_tree.scatter(x, yy, s=size, c=[color], edgecolors=edgecolor,
                    linewidths=linewidth, zorder=3)

    label = f"({node[0][0]},{node[0][1]};{node[1][0]},{node[1][1]})"
    ax_tree.text(x, yy - 0.75, label, ha='center', va='top',
                 fontsize=22, zorder=4, family='monospace')

# Axis formatting
ax_tree.set_xlim(-1.0, K * 3.5 + 1.5)
y_min = min(yy for _, yy in pos.values()) - 2
y_max = max(yy for _, yy in pos.values()) + 2
ax_tree.set_ylim(y_min, y_max)
for k in range(K + 1):
    ax_tree.axvline(k * 3.5, color='gray', alpha=0.15, linewidth=1.0, zorder=0)
    ax_tree.text(k * 3.5, y_max - 0.3,
                 f"$k = {k}$\n$|{{\\rm levels}}| = {len(levels[k])}$",
                 ha='center', va='top', fontsize=32, fontweight='bold')
ax_tree.set_title(
    f"Reachable count-tuple DAG: Ramsey-like DP, $K={K}$ epochs, $J={J}$ delays, $n=1$ shot/epoch\n"
    f"Node label: $(N_1,M_1;N_2,M_2)$.  Color = posterior mean $\\hat x$.  "
    f"Size $\\propto$ precision $1/\\sigma$.  Red ring = multi-parent (memoization collapse).",
    fontsize=28)
ax_tree.set_xticks([])
ax_tree.set_yticks([])
for spine in ax_tree.spines.values():
    spine.set_visible(False)

# Legend
legend_elements = [
    Line2D([0], [0], color=ACTION_COLOR[0], linestyle=OUTCOME_STYLE[0],
           lw=4, label=r"$\tau_1$ (short), $y=0$"),
    Line2D([0], [0], color=ACTION_COLOR[0], linestyle=OUTCOME_STYLE[1],
           lw=4, label=r"$\tau_1$ (short), $y=1$"),
    Line2D([0], [0], color=ACTION_COLOR[1], linestyle=OUTCOME_STYLE[0],
           lw=4, label=r"$\tau_2$ (long), $y=0$"),
    Line2D([0], [0], color=ACTION_COLOR[1], linestyle=OUTCOME_STYLE[1],
           lw=4, label=r"$\tau_2$ (long), $y=1$"),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
           markeredgecolor='red', markeredgewidth=4, markersize=24,
           lw=0, label='merged (>1 parent path)'),
]
ax_tree.legend(handles=legend_elements, loc='lower left', fontsize=22, framealpha=0.95)

# Summary text
naive_tree = sum((J * 2) ** k for k in range(K + 1))
actual_dag = sum(len(L) for L in levels)
ax_tree.text(K * 3.5 + 0.2, y_min + 0.5,
             f"naive tree: $\\sum_k (|A|)^k = {naive_tree}$\n"
             f"count-tuple DAG: {actual_dag}\n"
             f"compression: {naive_tree / actual_dag:.1f}×",
             fontsize=22, family='monospace', va='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', linewidth=1.5))

# -----------------------------------------------------------------------------
# Right small panels: four representative posteriors
# -----------------------------------------------------------------------------
rep_nodes = [
    empty,                              # prior
    ((1, 1), (1, 0)),                   # after τ_1-hit, τ_2-miss
    ((2, 2), (1, 1)),                   # terminal: 2 τ_1-hits, 1 τ_2-hit
    ((1, 0), (2, 1)),                   # terminal: τ_1-miss, 1 of 2 τ_2 hits
]

for i, node in enumerate(rep_nodes):
    ax_p = plt.subplot2grid((4, 5), (i, 4), rowspan=1, colspan=1)
    p = posterior(node)
    mean, sigma = stats[node]
    color_val = (mean - min_mean) / (max_mean - min_mean + 1e-12)
    color = cm.plasma(color_val)
    ax_p.plot(X, p, color='#222', linewidth=2.5)
    ax_p.fill_between(X, p, alpha=0.40, color=color)
    ax_p.axvline(mean, color='red', linestyle='--', linewidth=2.0, alpha=0.85)
    ax_p.axvline(PRIOR_MU, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax_p.set_xlim(0, 1)
    ax_p.set_xlabel(r"$x$", fontsize=22)
    ax_p.set_ylabel(r"$p(x \mid \mathbf{n})$", fontsize=20)
    tup_str = f"$({node[0][0]},{node[0][1]};{node[1][0]},{node[1][1]})$"
    label = "prior" if node == empty else tup_str
    ax_p.set_title(f"{label}     $\\hat x={mean:.3f}$, $\\sigma={sigma:.3f}$",
                   fontsize=20)
    ax_p.tick_params(labelsize=16)
    ax_p.grid(True, alpha=0.25)

plt.tight_layout()
out_path = "f:/My Drive/BFIMGaussian/dev/belief_tree.png"
plt.savefig(out_path, dpi=140, bbox_inches='tight')
print(f"Saved: {out_path}")
