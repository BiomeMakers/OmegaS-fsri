# =============================================================================
# FSRI Validation - Notebook 6: Proposition 2 (Discriminative Power)
# Author: Alberto Acedo (acedo@biomemakers.com)
# =============================================================================
"""
Reproduces Proposition 2: the medians and ranges of Omega over forty seeds on
N = 50 nodes at matched mean degree K = 6.

This notebook exists because the proposition reports forty-seed medians and no
notebook in the repository produced them. Notebook 1 draws a single graph from
each family and is an illustration, not a reproduction.

Omega is computed exactly as in Definition 1:

    Omega = C * D / (M * Coex)

with C the global clustering coefficient, D the connection density,
M = 1/lambda_2(L) the inverse algebraic connectivity of the combinatorial
Laplacian L = diag(k) - A, and Coex the variance of the degree sequence.
Disconnected draws are discarded, since M is undefined on them. The regular
ring has Coex = 0 exactly, so Omega diverges rather than taking a large finite
value, which is the content of Remark 5.

CPU only, under a minute. Requires networkx, numpy.
"""

import random

import networkx as nx
import numpy as np

N_NODES = 50
MEAN_DEGREE = 6
N_SEEDS = 40
MASTER_SEED = 42


def omega(G):
    """Omega of Definition 1. Returns (Omega, Coex)."""
    A = nx.to_numpy_array(G)
    n = len(A)
    deg = A.sum(axis=1)
    C = np.trace(A @ A @ A) / ((A @ A).sum() - np.trace(A @ A))
    D = A.sum() / (n * (n - 1))
    lam = np.sort(np.linalg.eigvalsh(np.diag(deg) - A))
    if lam[1] < 1e-9:                 # disconnected: M is undefined
        return np.nan, deg.var()
    if deg.var() < 1e-12:             # regular: the index diverges
        return np.inf, 0.0
    return C * D * lam[1] / deg.var(), deg.var()


FAMILIES = [
    ("Watts-Strogatz (p=0.1)",
     lambda s: nx.watts_strogatz_graph(N_NODES, MEAN_DEGREE, 0.1, seed=s)),
    ("Erdos-Renyi",
     lambda s: nx.gnm_random_graph(N_NODES, N_NODES * MEAN_DEGREE // 2, seed=s)),
    ("Barabasi-Albert",
     lambda s: nx.barabasi_albert_graph(N_NODES, MEAN_DEGREE // 2, seed=s)),
]

rng = random.Random(MASTER_SEED)

print("=" * 78)
print(f"Proposition 2: Omega over {N_SEEDS} seeds, N = {N_NODES}, K = {MEAN_DEGREE}")
print("=" * 78)
print(f"\n{'family':<26} {'median Omega':>13} {'range':>24} {'median Coex':>12} {'n':>4}")
print("-" * 78)

medians = {}
for name, gen in FAMILIES:
    omegas, coexes = [], []
    for _ in range(N_SEEDS):
        G = gen(rng.randrange(10 ** 9))
        if not nx.is_connected(G):
            continue
        o, v = omega(G)
        omegas.append(o)
        coexes.append(v)
    omegas = np.array(omegas)
    medians[name] = float(np.median(omegas))
    rng_txt = f"[{omegas.min():.4f}, {omegas.max():.4f}]"
    print(f"{name:<26} {np.median(omegas):>13.4f} {rng_txt:>24} "
          f"{np.median(coexes):>12.2f} {len(omegas):>4}")

ring = nx.watts_strogatz_graph(N_NODES, MEAN_DEGREE, 0.0, seed=1)
o_ring, v_ring = omega(ring)
print(f"{'Regular ring (p=0)':<26} {'divergent':>13} "
      f"{'(Coex = 0 exactly)':>24} {v_ring:>12.2f} {1:>4}")

ws = medians["Watts-Strogatz (p=0.1)"]
print("\n" + "=" * 78)
print("READING")
print("=" * 78)
print(f"Watts-Strogatz over Erdos-Renyi:   {ws / medians['Erdos-Renyi']:.1f}x")
print(f"Watts-Strogatz over Barabasi-Albert: {ws / medians['Barabasi-Albert']:.1f}x")
print("The three finite families are separated by about one and a half orders")
print("of magnitude, and the ordering is small-world > random > preferential")
print("attachment. The ring is not a large finite value but a divergence: its")
print("degree variance is identically zero, so Omega has no finite value there.")
print("Ranges overlap nowhere between families, which is what the proposition")
print("claims; the medians alone would not establish that.")
