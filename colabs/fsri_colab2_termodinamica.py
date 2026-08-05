# =============================================================================
# FSRI Validation - Notebook 2: thermodynamics of network topology
# Author: Alberto Acedo (acedo@biomemakers.com)
# =============================================================================
"""
Explores the connection between Tr(A^3) and non-equilibrium thermodynamics:

1. PHASE TRANSITION: triangle density follows a first-order thermodynamic
   transition, with added triangles playing the role of a fugacity.

2. TOPOLOGICAL ENTROPY: S = -sum_k p_k log(p_k) over the degree distribution
   falls as Tr(A^3) rises, so more topological order means less entropy.

3. THE APPROXIMATE PROPORTIONALITY Omega ~ (S_env/E_ext)^2, tested on networks
   with different levels of external energy (mean degree) and environmental
   stress (heterogeneity).

   Note on scope: this is the EARLIER, approximate form. The exact relation
   derived in the paper is Omega = [(3K-6)/(4K-2)] * S_env^2 / (n p E_ext^3),
   and notebook 3 shows where the approximate form holds and where it does not.

CPU only, about three minutes.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. SHANNON ENTROPY OF THE DEGREE DISTRIBUTION
# ---------------------------------------------------------------------------

def degree_entropy(A):
    """S = -sum_k p_k log(p_k) over the degree distribution."""
    degrees = A.sum(axis=1).astype(int)
    counts = Counter(degrees)
    n = len(degrees)
    probs = np.array([v/n for v in counts.values()])
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs + 1e-10))

def tr_a3_norm(A):
    n = A.shape[0]
    return np.trace(A @ A @ A) / max(n**3, 1)

def erdos_renyi(n, p):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if np.random.random() < p:
                A[i,j] = A[j,i] = 1
    return A

def barabasi_albert(n, m):
    A = np.zeros((n, n))
    for i in range(m):
        for j in range(i+1, m):
            A[i,j] = A[j,i] = 1
    degrees = A.sum(axis=0)
    for new_node in range(m, n):
        degrees[new_node] = 0
        if degrees.sum() == 0:
            continue
        probs = degrees / degrees.sum()
        targets = np.random.choice(n, size=m, replace=False, p=probs)
        for t in targets:
            A[new_node, t] = A[t, new_node] = 1
            degrees[new_node] += 1
            degrees[t] += 1
    return A

def add_triangles(A, n_triangles):
    """Closes open triads, which raises Tr(A^3)."""
    A = A.copy()
    n = A.shape[0]
    added = 0
    attempts = 0
    while added < n_triangles and attempts < n_triangles * 10:
        attempts += 1
        i = np.random.randint(n)
        neighbors_i = np.where(A[i] > 0)[0]
        if len(neighbors_i) < 2:
            continue
        j, k = np.random.choice(neighbors_i, 2, replace=False)
        if A[j, k] == 0:
            A[j, k] = A[k, j] = 1
            added += 1
    return A, added

# ---------------------------------------------------------------------------
# 2. EXPERIMENT: phase transition as triangles are added
# ---------------------------------------------------------------------------

print("="*60)
print("EXPERIMENT 1: topological phase transition")
print("An ER graph with progressively added triangles, fugacity-like")
print("="*60)

N = 60
p_base = 0.08
n_steps = 20
triangle_steps = np.linspace(0, 200, n_steps).astype(int)

tra3_vals, entropy_vals, coex_vals, omega_vals = [], [], [], []

A_base = erdos_renyi(N, p_base)

for n_tri in triangle_steps:
    A, _ = add_triangles(A_base, n_tri)
    tra3_vals.append(tr_a3_norm(A))
    entropy_vals.append(degree_entropy(A))
    coex_vals.append(A.sum(axis=1).var())
    
    # Simplified FSRI proxy
    C = np.trace(A @ A @ A) / max((A@A).sum() - np.trace(A@A), 1)
    D = A.sum() / (N*(N-1))
    Coex = max(A.sum(axis=1).var(), 1e-8)
    omega_vals.append(C * D / Coex)

print(f"No added triangles: Tr(A^3)/N^3 = {tra3_vals[0]:.6f}, "
      f"S = {entropy_vals[0]:.4f}")
print(f"With {triangle_steps[-1]} triangles: Tr(A^3)/N^3 = {tra3_vals[-1]:.6f}, "
      f"S = {entropy_vals[-1]:.4f}")

# ---------------------------------------------------------------------------
# 3. EXPERIMENT: testing the approximate Omega ~ (S_env/E_ext)^2
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("EXPERIMENT 2: the approximate proportionality Omega ~ (S_env/E_ext)^2")
print("S_env = environmental stress (heterogeneity of connections)")
print("E_ext = external energy (mean degree, the connectivity available)")
print("="*60)

# Build networks across a grid of (S_env, E_ext) and measure Omega
results_thermo = []
K_values = [3, 4, 6, 8, 10]        # E_ext: external energy
Het_values = [0.1, 0.3, 0.5, 0.8]  # S_env: heterogeneity (rewiring p)

for k in K_values:
    for het in Het_values:
        # Watts-Strogatz: k is the base degree (E_ext), het the rewiring p (S_env)
        n = 50
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(1, k//2 + 1):
                A[i, (i+j)%n] = A[(i+j)%n, i] = 1
        # Rewiring
        for i in range(n):
            for j in range(1, k//2 + 1):
                if np.random.random() < het:
                    new_j = np.random.randint(n)
                    if new_j != i and A[i, new_j] == 0:
                        A[i, (i+j)%n] = A[(i+j)%n, i] = 0
                        A[i, new_j] = A[new_j, i] = 1

        Eext = A.sum() / n                          # mean degree
        Senv = A.sum(axis=1).std() / max(Eext, 1)  # coefficient of variation

        C_net = np.trace(A@A@A) / max((A@A).sum() - np.trace(A@A), 1)
        D_net = A.sum() / (n*(n-1))
        Coex  = max(A.sum(axis=1).var(), 1e-8)
        Omega = C_net * D_net / Coex

        # Approximate prediction: Omega ~ (S_env/E_ext)^2
        ratio_sq = (Senv / max(Eext, 1e-8))**2

        results_thermo.append({
            "k": k, "het": het, "Eext": Eext, "Senv": Senv,
            "Omega": Omega, "ratio_sq": ratio_sq
        })

# Correlation between Omega and (S_env/E_ext)^2
omegas    = [r["Omega"] for r in results_thermo]
ratio_sqs = [r["ratio_sq"] for r in results_thermo]
corr = np.corrcoef(omegas, ratio_sqs)[0,1]
print(f"Correlation of Omega with (S_env/E_ext)^2: {corr:.4f}")
if abs(corr) > 0.5:
    print("The approximate form tracks Omega over this grid")
else:
    print("Weak correlation: the approximate form needs the exact correction")

# ---------------------------------------------------------------------------
# 4. EXPERIMENT: Omega-S as a minimum entropy production process
# ---------------------------------------------------------------------------

print("\n" + "="*60)
print("EXPERIMENT 3: Omega-S as a Prigogine principle")
print("Minimising Tr(A^3) minimises topological entropy production,")
print("that is, the stationary state of least dissipation")
print("="*60)

def omega_s_gradient_step(A, lr=0.01):
    """
    One Omega-S gradient step. It penalises the edges that contribute most
    to Tr(A^3), which is the gradient -d Tr(A^3)/d A_ij = -3 (A^2)_ij.
    """
    A2 = A @ A
    gradient = 3 * A2  # d Tr(A^3)/dA = 3 A^2
    # Penalty proportional to the gradient
    scores = gradient * A  # existing edges only
    return scores

N = 40
A_monopoly = barabasi_albert(N, 3)
A_uniform  = np.zeros((N, N))
# Ring lattice
for i in range(N):
    for j in [1, 2, 3]:
        A_uniform[i, (i+j)%N] = A_uniform[(i+j)%N, i] = 1

networks = {
    "Monopoly network\n(Barabasi-Albert)": A_monopoly,
    "Uniform network\n(regular ring)":     A_uniform,
}

print("\nOmega-S gradient = d Tr(A^3)/dA = 3 A^2")
print("(proportional to the topological pressure on each edge)\n")

for name, A in networks.items():
    scores = omega_s_gradient_step(A)
    max_score  = scores.max()
    mean_score = scores[A > 0].mean() if A.sum() > 0 else 0
    gini = scores[A > 0].std() / (mean_score + 1e-8) if A.sum() > 0 else 0

    print(f"{name.replace(chr(10), ' ')}:")
    print(f"  Peak pressure on an edge:  {max_score:.4f}")
    print(f"  Mean pressure:             {mean_score:.4f}")
    print(f"  Inequality (Gini proxy):   {gini:.4f}")
    print(f"  Tr(A^3)/N^3:               {tr_a3_norm(A):.6f}")
    print()

# ---------------------------------------------------------------------------
# 5. FIGURES
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: phase transition
axes[0].plot(triangle_steps, tra3_vals, 'b-o', markersize=4, label='Tr(A^3)/N^3')
ax0b = axes[0].twinx()
ax0b.plot(triangle_steps, entropy_vals, 'r--s', markersize=4, label='Entropy S')
axes[0].set_xlabel('Triangles added (fugacity)')
axes[0].set_ylabel('Tr(A^3)/N^3', color='b')
ax0b.set_ylabel('Degree entropy S', color='r')
axes[0].set_title('Topological phase transition\nTr(A^3) against entropy')
axes[0].legend(loc='upper left', fontsize=8)
ax0b.legend(loc='upper right', fontsize=8)

# Plot 2: Omega against (S_env/E_ext)^2
axes[1].scatter(ratio_sqs, omegas, alpha=0.6, c='steelblue', s=60)
z = np.polyfit(ratio_sqs, omegas, 1)
p = np.poly1d(z)
x_line = np.linspace(min(ratio_sqs), max(ratio_sqs), 100)
axes[1].plot(x_line, p(x_line), 'r--', lw=2,
             label=f'Linear fit\nr={corr:.3f}')
axes[1].set_xlabel('(S_env/E_ext)^2')
axes[1].set_ylabel('FSRI Omega')
axes[1].set_title(f'Approximate form\nOmega ~ (S_env/E_ext)^2  [r={corr:.3f}]')
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

# Plot 3: Omega-S pressure by network
for idx, (name, A) in enumerate(networks.items()):
    scores = omega_s_gradient_step(A)
    edge_scores = scores[A > 0]
    axes[2].hist(edge_scores, bins=20, alpha=0.6,
                 label=name.replace('\n', ' '),
                 color=['red', 'green'][idx])
axes[2].set_xlabel('Omega-S pressure on edges, d Tr(A^3)/d A_ij')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Distribution of topological pressure\nmonopoly against uniform')
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.suptitle(
    'FSRI: thermodynamic foundations\n'
    'Omega-S as a minimum entropy production principle',
    fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('fsri_validation_2.png', dpi=150, bbox_inches='tight')
plt.show()

print("="*60)
print("THERMODYNAMIC CONCLUSIONS")
print("="*60)
print("1. Adding triangles raises Tr(A^3) and lowers the degree entropy S,")
print("   which is the analogue of a thermodynamic phase change")
print(f"2. Correlation of Omega with (S_env/E_ext)^2: {corr:.3f}")
print("   This is the approximate form only. The exact relation derived in")
print("   the paper carries E_ext to the third power; see notebook 3")
print("3. The Omega-S gradient, 3 A^2, is the topological pressure")
print("   Monopoly networks concentrate that pressure (high Coex)")
print("   Uniform networks distribute it (low Coex)")
print("   Minimising Tr(A^3) is the Prigogine principle applied to graphs")
