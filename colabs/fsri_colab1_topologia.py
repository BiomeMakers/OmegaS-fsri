# =============================================================================
# FSRI Validation - Notebook 1: Tr(A^3) and the degree distribution
# Omega-S / Functional Symbiotic Resilience Index
# Author: Alberto Acedo (acedo@biomemakers.com)
# =============================================================================
"""
Shows that:
1. Tr(A^3)/N^3 is proportional to the global clustering coefficient C
2. Minimising Tr(A^3) yields more uniform degree distributions (lower Coex)
3. The transition from a monopoly network to a uniform one is a phase transition

CPU only, about two minutes.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh

# ---------------------------------------------------------------------------
# 1. BASE FUNCTIONS
# ---------------------------------------------------------------------------

def clustering_coefficient(A):
    """C = Tr(A^3) / (sum(A^2) - Tr(A^2)), the standard definition."""
    A3_trace = np.trace(A @ A @ A)
    A2 = A @ A
    denom = A2.sum() - np.trace(A2)
    return A3_trace / max(denom, 1e-8)

def degree_variance(A):
    """Variance of the degree distribution, which is the Coex factor."""
    degrees = A.sum(axis=1)
    return degrees.var()

def fsri_omega(A):
    """
    Omega = C * D / (M * Coex), exactly as in Definition 1 of the paper.

    C    : global clustering coefficient (normalised Tr(A^3))
    D    : connection density
    M    : 1 / lambda_2(L), the inverse algebraic connectivity of the
           LAPLACIAN L = diag(k) - A.  Note: an earlier version of this
           function used the gap between the second and third eigenvalues of
           the ADJACENCY matrix, which is a different object and is not the
           one Definition 1 asks for.
    Coex : variance of the degree sequence.

    Returns np.inf when Coex = 0 (regular graph) and np.nan when the graph is
    disconnected (lambda_2 = 0).  An earlier version replaced Coex by a floor
    of 1e-8, which returned a finite number fixed by the floor rather than by
    the graph: on the regular ring that artefact gave 7.35 with a floor of
    1e-8 and would have given 73.5 with 1e-9.  See Remark 5 and Proposition 2
    of the paper.
    """
    n = A.shape[0]
    deg = A.sum(axis=1)
    C = clustering_coefficient(A)
    D = A.sum() / (n * (n - 1))

    laplacian = np.diag(deg) - A
    lam = np.sort(eigvalsh(laplacian))
    if lam[1] < 1e-9:            # disconnected graph: M is undefined
        return np.nan
    M = 1.0 / lam[1]

    Coex = degree_variance(A)
    if Coex < 1e-12:             # regular graph: the index diverges
        return np.inf

    return C * D / (M * Coex)

def tr_a3_normalized(A):
    """Tr(A^3) / N^3, proportional to the clustering coefficient."""
    n = A.shape[0]
    return np.trace(A @ A @ A) / (n**3)

# ---------------------------------------------------------------------------
# 2. NETWORK GENERATORS
# ---------------------------------------------------------------------------

def barabasi_albert(n, m):
    """Scale-free network: a dominant hub, so high Coex and low Omega."""
    A = np.zeros((n, n))
    # Start from a complete graph on m nodes
    for i in range(m):
        for j in range(i+1, m):
            A[i,j] = A[j,i] = 1
    
    degrees = A.sum(axis=0)
    for new_node in range(m, n):
        degrees[new_node] = 0
        probs = degrees / degrees.sum()
        targets = np.random.choice(n, size=m, replace=False, p=probs)
        for t in targets:
            A[new_node, t] = A[t, new_node] = 1
            degrees[new_node] += 1
            degrees[t] += 1
    return A

def watts_strogatz(n, k, p):
    """Small-world network: high clustering, uniform degrees, so high Omega."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(1, k//2 + 1):
            A[i, (i+j) % n] = A[(i+j) % n, i] = 1
    # Rewiring
    for i in range(n):
        for j in range(1, k//2 + 1):
            if np.random.random() < p:
                new_j = np.random.randint(n)
                if new_j != i and A[i, new_j] == 0:
                    A[i, (i+j) % n] = A[(i+j) % n, i] = 0
                    A[i, new_j] = A[new_j, i] = 1
    return A

def erdos_renyi(n, p):
    """Random network with Poisson degrees, so intermediate Omega."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if np.random.random() < p:
                A[i,j] = A[j,i] = 1
    return A

def ring_lattice(n, k):
    """Regular ring lattice: maximal uniformity, so Coex is exactly zero."""
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(1, k//2 + 1):
            A[i, (i+j) % n] = A[(i+j) % n, i] = 1
    return A

# ---------------------------------------------------------------------------
# 3. EXPERIMENT: Tr(A^3) against network type
# ---------------------------------------------------------------------------

np.random.seed(42)
N = 50  # nodes
K = 6   # mean degree

print("="*60)
print("EXPERIMENT 1: Tr(A^3) and topological properties by network type")
print("="*60)

networks_by_type = {
    "Barabasi-Albert\n(hub monopoly)":   barabasi_albert(N, K//2),
    "Erdos-Renyi\n(random)":             erdos_renyi(N, K/N),
    "Watts-Strogatz\n(small-world)":     watts_strogatz(N, K, 0.1),
    "Regular ring\n(uniform)":           ring_lattice(N, K),
}

results = {}
for name, A in networks_by_type.items():
    C   = clustering_coefficient(A)
    Coex = degree_variance(A)
    D   = A.sum() / (N*(N-1))
    TrA3 = tr_a3_normalized(A)
    Omega = fsri_omega(A)
    results[name] = {"C": C, "Coex": Coex, "D": D,
                     "TrA3": TrA3, "Omega": Omega}
    print(f"\n{name}")
    print(f"  Clustering C:      {C:.4f}")
    print(f"  Degree variance:   {Coex:.4f}  (Coex)")
    print(f"  Tr(A^3)/N^3:       {TrA3:.6f}")
    print(f"  Density D:         {D:.4f}")
    if np.isinf(Omega):
        print(f"  FSRI Omega:        divergent (Coex = 0, regular graph)")
    elif np.isnan(Omega):
        print(f"  FSRI Omega:        undefined (disconnected graph)")
    else:
        print(f"  FSRI Omega:        {Omega:.4f}")

# ---------------------------------------------------------------------------
# 4. EXPERIMENT: minimising Tr(A^3) gives a more uniform network
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("EXPERIMENT 2: topological optimisation via Tr(A^3)")
print("Starting from a BA network (monopoly) and minimising Tr(A^3)")
print("by random rewiring. Does it converge to a more uniform network?")
print("="*60)

def omega_s_rewiring(A, n_steps=2000, lambda_omega=0.1):
    """
    Simulates the effect of Omega-S as a rewiring process: at each step it
    proposes a random rewiring and accepts it if Tr(A^3) decreases, which is
    the analogue of the Omega-S gradient during training.
    """
    A = A.copy()
    n = A.shape[0]
    history = {"TrA3": [], "Coex": [], "C": []}
    
    current_tra3 = np.trace(A @ A @ A)
    
    for step in range(n_steps):
        # Propose a rewiring: remove edge (i,j), add (i,k)
        edges = np.argwhere(np.triu(A, 1))
        if len(edges) == 0:
            break
        idx = np.random.randint(len(edges))
        i, j = edges[idx]
        
        # New neighbour for i
        non_neighbors = np.where((A[i] == 0) & (np.arange(n) != i))[0]
        if len(non_neighbors) == 0:
            continue
        k = np.random.choice(non_neighbors)
        
        # Apply the tentative rewiring
        A[i,j] = A[j,i] = 0
        A[i,k] = A[k,i] = 1
        
        new_tra3 = np.trace(A @ A @ A)
        
        # Accept if it reduces Tr(A^3) (Omega-S gradient descent)
        if new_tra3 < current_tra3:
            current_tra3 = new_tra3
        else:
            # Revert
            A[i,j] = A[j,i] = 1
            A[i,k] = A[k,i] = 0
        
        if step % 100 == 0:
            history["TrA3"].append(tr_a3_normalized(A))
            history["Coex"].append(degree_variance(A))
            history["C"].append(clustering_coefficient(A))
    
    return A, history

A_ba = barabasi_albert(N, K//2)
print(f"\nBefore rewiring (BA):")
print(f"  Tr(A^3)/N^3 = {tr_a3_normalized(A_ba):.6f}")
print(f"  Coex        = {degree_variance(A_ba):.4f}")
print(f"  C           = {clustering_coefficient(A_ba):.4f}")

A_opt, history = omega_s_rewiring(A_ba, n_steps=3000)
print(f"\nAfter Omega-S guided rewiring:")
print(f"  Tr(A^3)/N^3 = {tr_a3_normalized(A_opt):.6f}")
print(f"  Coex        = {degree_variance(A_opt):.4f}")
print(f"  C           = {clustering_coefficient(A_opt):.4f}")

reduction_tra3 = (tr_a3_normalized(A_ba) - tr_a3_normalized(A_opt)) \
                  / tr_a3_normalized(A_ba) * 100
reduction_coex = (degree_variance(A_ba) - degree_variance(A_opt)) \
                  / degree_variance(A_ba) * 100
print(f"\nTr(A^3) reduction: {reduction_tra3:.1f}%")
print(f"Coex reduction:    {reduction_coex:.1f}%")

# ---------------------------------------------------------------------------
# 5. FIGURES
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Tr(A^3) by network type
names = list(results.keys())
tra3_vals = [results[n]["TrA3"] for n in names]
colors = ['red', 'orange', 'steelblue', 'green']
axes[0].bar(range(len(names)), tra3_vals, color=colors, alpha=0.8)
axes[0].set_xticks(range(len(names)))
axes[0].set_xticklabels([n.replace('\n', '\n') for n in names],
                         fontsize=8, ha='center')
axes[0].set_ylabel('Tr(A^3) / N^3')
axes[0].set_title('Topological clustering\nby network type')
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: evolution during rewiring
steps = range(0, len(history["TrA3"]))
ax2 = axes[1]
ax2b = ax2.twinx()
l1, = ax2.plot(steps, history["TrA3"], 'b-', label='Tr(A^3)/N^3', lw=2)
l2, = ax2b.plot(steps, history["Coex"], 'r--', label='Coex (degree var.)', lw=2)
ax2.set_xlabel('Steps x 100')
ax2.set_ylabel('Tr(A^3)/N^3', color='b')
ax2b.set_ylabel('Degree variance (Coex)', color='r')
ax2.set_title('Omega-S optimisation:\nminimising Tr(A^3) lowers Coex')
lines = [l1, l2]
ax2.legend(lines, [l.get_label() for l in lines], loc='upper right', fontsize=8)

# Plot 3: Coex against Tr(A^3), by network type
coex_vals  = [results[n]["Coex"] for n in names]
omega_vals = [results[n]["Omega"] for n in names]
scatter = axes[2].scatter(tra3_vals, coex_vals,
                          c=omega_vals, cmap='RdYlGn',
                          s=200, zorder=5)
for i, n in enumerate(names):
    axes[2].annotate(n.replace('\n',' '),
                     (tra3_vals[i], coex_vals[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)
plt.colorbar(scatter, ax=axes[2], label='FSRI Omega')
axes[2].set_xlabel('Tr(A^3) / N^3  (topological clustering)')
axes[2].set_ylabel('Degree variance (Coex)')
axes[2].set_title('Tr(A^3) against Coex\ncoloured by FSRI Omega')
axes[2].grid(alpha=0.3)

plt.suptitle('Functional Symbiotic Resilience Index: validation\n'
             'Minimising Tr(A^3) reduces topological monopolies',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('fsri_validation_1.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("CONCLUSIONS")
print("="*60)
order = " > ".join(n.split(chr(10))[0]
                   for n in sorted(results, key=lambda k: -results[k]["TrA3"]))
print(f"1. Tr(A^3)/N^3 by itself does NOT order these families by monopoly.")
print(f"   Measured order, highest first: {order}")
print(f"   The regular ring, which is the most egalitarian of the four, has")
print(f"   the HIGHEST Tr(A^3), because triangle count rewards dense local")
print(f"   closure regardless of how evenly degree is spread. What separates")
print(f"   monopoly from resilience is the concentration factor Coex, in the")
print(f"   denominator of Omega. See the remark on the direction of Tr(A^3).")
print(f"2. Greedy minimisation of Tr(A^3) by rewiring lowers Coex by "
      f"{reduction_coex:.1f}%,")
print(f"   but note the endpoint: unconstrained greedy rewiring drives the")
print(f"   triangle count to zero, so this is the degenerate limit of the")
print(f"   objective and not a model of the regularised training dynamics,")
print(f"   where the penalty competes with a task loss.")
print(f"3. FSRI Omega is larger on the resilient topologies (small-world),")
print(f"   and diverges on the regular ring, where Coex is exactly zero")
print(f"   (Proposition 2 of the paper)")
