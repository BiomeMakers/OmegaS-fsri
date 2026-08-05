# =============================================================================
# FSRI Validation - Notebook 3: ORC against Tr(A^3), and the exact WS formula
# Author: Alberto Acedo (acedo@biomemakers.com)
# =============================================================================
"""
Checks numerically:
1. The correlation between the local ORC kappa(x,y) and each edge's
   contribution to Tr(A^3)
2. The exact closed-form Omega(K,p) on Watts-Strogatz graphs
3. The corrected relation Omega * p ~ (S_env/E_ext)^2 * (1-p)

Requirements:
  pip install networkx numpy scipy matplotlib
CPU only, about two minutes.
"""

import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import networkx as nx

# Both generators have to be seeded. numpy alone is not enough: the networkx
# graph generators used below draw from Python's `random` module, so seeding
# only numpy left this notebook non-reproducible between runs.
np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# 1. ORC COMPUTED DIRECTLY, WITHOUT AN EXTERNAL LIBRARY
# ---------------------------------------------------------------------------

def ollivier_ricci_curvature(G, x, y):
    """
    ORC of edge (x,y) with the uniform distribution over neighbours,
    kappa(x,y) = 1 - W1(mu_x, mu_y) / d(x,y).
    On unweighted graphs W1 is the earth mover distance between two uniform
    distributions.

    Note: this function uses the Jost-Liu style approximation, which is fast
    but imprecise on heterogeneous graphs. Notebook 4b computes the exact
    Wasserstein-1 curvature and is the one that generates Table 3 of the paper.
    """
    nx_neighbors = set(G.neighbors(x)) - {y}
    ny_neighbors = set(G.neighbors(y)) - {x}
    
    deg_x = G.degree(x)
    deg_y = G.degree(y)
    
    if deg_x == 0 or deg_y == 0:
        return 0.0
    
    # Common neighbours contribute positively to kappa
    common = nx_neighbors & ny_neighbors

    # Jost-Liu style approximation for near-regular graphs:
    # kappa(x,y) ~ |N(x) n N(y)| * (1/deg_x + 1/deg_y) - 1 + 1/deg_x + 1/deg_y
    # It recovers the Lin-Lu-Yau expression on regular graphs.

    triangles = len(common)

    kappa = triangles * (1/deg_x + 1/deg_y) - 1 + 1/max(deg_x,1) + 1/max(deg_y,1)
    # The theoretical maximum is 2/deg, reached when every neighbour is common.

    return kappa

def triangle_contribution(A, x, y):
    """Contribution of edge (x,y) to Tr(A^3)."""
    # (A^2)_{xy} counts paths of length two between x and y, that is, the
    # number of common neighbours.
    A2 = A @ A
    return A2[x, y]

# ---------------------------------------------------------------------------
# 2. EXPERIMENT: correlation of ORC with the contribution to Tr(A^3)
# ---------------------------------------------------------------------------

print("="*65)
print("EXPERIMENT 1: ORC against contribution to Tr(A^3)")
print("="*65)

network_types = {
    "Watts-Strogatz (k=6, p=0.1)": nx.watts_strogatz_graph(50, 6, 0.1),
    "Barabasi-Albert (m=3)":        nx.barabasi_albert_graph(50, 3),
    "Erdos-Renyi (p=0.12)":         nx.erdos_renyi_graph(50, 0.12),
}

for name, G in network_types.items():
    A = nx.to_numpy_array(G)
    
    kappas, tri_contribs = [], []
    for x, y in G.edges():
        k = ollivier_ricci_curvature(G, x, y)
        tc = triangle_contribution(A, x, y)
        kappas.append(k)
        tri_contribs.append(tc)
    
    if len(kappas) > 2:
        corr, pval = pearsonr(kappas, tri_contribs)
        kappa_plus = [k for k in kappas if k > 0]
        frac_positive = len(kappa_plus) / len(kappas)
        
        print(f"\n{name}:")
        print(f"  Correlation of kappa with common neighbours: {corr:.4f} (p={pval:.4f})")
        print(f"  Fraction of edges with kappa > 0:            {frac_positive:.2%}")
        print(f"  Tr(A^3)/N^3:                                 {np.trace(A@A@A)/50**3:.6f}")
        print(f"  Relation to sum of positive kappa: {'confirmed' if corr > 0.7 else 'partial'}")

# ---------------------------------------------------------------------------
# 3. THE EXACT WS FORMULA: Omega(K,p), closed form against simulation
# ---------------------------------------------------------------------------

print("\n" + "="*65)
print("EXPERIMENT 2: exact Omega(K,p), closed form against simulation")
print("="*65)

def C_ws_exact(K, p):
    """C(K,p) = (3K-6)/(4K-2) * (1-p)^3, from Barrat and Weigt (2000)."""
    return (3*K - 6) / (4*K - 2) * (1-p)**3

def var_k_ws_approx(K, p):
    """Var(k) ~ K p (1-p), in the large-N limit."""
    return K * p * (1-p)

def omega_ws_analytical(K, p, N=100):
    C = C_ws_exact(K, p)
    D = K / N
    Coex = max(var_k_ws_approx(K, p), 1e-10)
    return C * D / Coex

def omega_ws_numerical(K, p, N=100, n_samples=200):
    """Omega measured on n_samples WS graphs and averaged.

    n_samples was 5, which is too few: the per-graph spread at small p made the
    reported correlation move by several thousandths between runs. At 200 the
    value is stable.
    """
    omegas = []
    for _ in range(n_samples):
        G = nx.watts_strogatz_graph(N, K, p)
        A = nx.to_numpy_array(G)
        degrees = A.sum(axis=1)
        C = np.trace(A@A@A) / max((A@A).sum() - np.trace(A@A), 1)
        D = A.sum() / (N*(N-1))
        Coex = max(degrees.var(), 1e-10)
        omegas.append(C * D / Coex)
    return np.mean(omegas)

K = 6
p_vals = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

print(f"\nK={K}, N=100")
print(f"{'p':>6} {'Omega closed':>14} {'Omega sim.':>12} {'Error %':>8}")
print("-"*44)

analytical_vals, numerical_vals = [], []
for p in p_vals:
    omega_a = omega_ws_analytical(K, p, N=100)
    omega_n = omega_ws_numerical(K, p, N=100)
    error = abs(omega_a - omega_n) / max(omega_n, 1e-10) * 100
    print(f"{p:>6.2f} {omega_a:>14.6f} {omega_n:>12.6f} {error:>8.1f}%")
    analytical_vals.append(omega_a)
    numerical_vals.append(omega_n)

corr_formula, _ = pearsonr(analytical_vals, numerical_vals)
print(f"\nCorrelation of closed form with simulation: {corr_formula:.4f}")
print(f"Omega = [(3K-6)/(4K-2)] (1-p)^2 / (N p): {'validated' if corr_formula > 0.95 else 'partially validated'}")

# ---------------------------------------------------------------------------
# 4. HOW Omega RELATES TO (S_env/E_ext)^2, AND IN WHICH DIRECTION
# ---------------------------------------------------------------------------

print("\n" + "="*65)
print("EXPERIMENT 3: the direction of the relation with (S_env/E_ext)^2")
print("="*65)

K_vals = [4, 6, 8, 10]
p_test = np.linspace(0.05, 0.8, 15)

omega_p_vals = []
ratio_vals = []

for K in K_vals:
    for p in p_test:
        Omega = omega_ws_analytical(K, p, N=100)
        Eext = K
        Senv = np.sqrt(var_k_ws_approx(K, p))
        
        omega_p_vals.append(Omega * p)
        ratio_vals.append((Senv/Eext)**2 * (1-p))

corr_corrected, pval_corrected = pearsonr(omega_p_vals, ratio_vals)
print(f"Correlation of Omega p with (S_env/E_ext)^2 (1-p): {corr_corrected:.4f} (p={pval_corrected:.6f})")
print("  (an earlier draft proposed this as the corrected relation; it is not,")
print("   and the near-zero correlation here is why)")

# The direct proportionality proposed in an earlier formulation of this
# framework, without the correction.
omega_direct = [omega_ws_analytical(K, p, N=100) for K in K_vals for p in p_test]
ratio_direct = [(np.sqrt(var_k_ws_approx(K,p))/K)**2 for K in K_vals for p in p_test]
corr_direct, _ = pearsonr(omega_direct, ratio_direct)
print(f"Correlation of Omega with (S_env/E_ext)^2 [direct form]: {corr_direct:.4f}")
print(f"\nBy strength of association, |r| = {abs(corr_corrected):.4f} against "
      f"{abs(corr_direct):.4f}:")
print(f"  the p(1-p) form carries essentially NO association, while the direct")
print(f"  form carries a clear NEGATIVE one. Comparing signed correlations would")
print(f"  wrongly call the first one better.")

# The exact identity, obtained by substituting Var(k) = K p (1-p) and E_ext = K
# into the closed form. It is exact, not a fit.
exact = [ (1/(100*K)) * ((3*K-6)/(4*K-2)) * (1-p)**3
          / ((np.sqrt(var_k_ws_approx(K,p))/K)**2)
          for K in K_vals for p in p_test ]
omega_all = [omega_ws_analytical(K, p, N=100) for K in K_vals for p in p_test]
max_rel = max(abs(a-b)/a for a, b in zip(omega_all, exact))
print(f"\nExact identity  Omega = [1/(nK)] [(3K-6)/(4K-2)] (1-p)^3 / (S_env/E_ext)^2")
print(f"  maximum relative error against the closed form: {max_rel:.2e}")
print(f"  The ratio sits in the DENOMINATOR, so the relation is inverse. That is")
print(f"  what produces the negative correlation above.")

# ---------------------------------------------------------------------------
# 5. FIGURES
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: ORC against contribution to Tr(A^3) on a WS graph
G_ws = nx.watts_strogatz_graph(60, 6, 0.1)
A_ws = nx.to_numpy_array(G_ws)
kappas_ws = [ollivier_ricci_curvature(G_ws, x, y) for x, y in G_ws.edges()]
tcs_ws = [triangle_contribution(A_ws, x, y) for x, y in G_ws.edges()]

colors = ['red' if k <= 0 else 'steelblue' for k in kappas_ws]
axes[0].scatter(kappas_ws, tcs_ws, c=colors, alpha=0.6, s=30)
axes[0].axvline(x=0, color='k', linestyle='--', alpha=0.5)
axes[0].set_xlabel('ORC kappa(x,y)')
axes[0].set_ylabel('Common neighbours')
axes[0].set_title('ORC against contribution to Tr(A^3)\n(WS, k=6, p=0.1)\nred: kappa<0 (bridges), blue: kappa>0 (triangles)')
r_ws, _ = pearsonr(kappas_ws, tcs_ws)
axes[0].text(0.05, 0.95, f'r = {r_ws:.3f}', transform=axes[0].transAxes,
             fontsize=11, va='top', fontweight='bold')
axes[0].grid(alpha=0.3)

# Plot 2: closed form against simulation
p_plot = np.linspace(0.05, 0.9, 30)
omega_a_plot = [omega_ws_analytical(6, p, 100) for p in p_plot]
axes[1].plot(p_plot, omega_a_plot, 'b-', lw=2, label='Closed form: (3K-6)/(4K-2)(1-p)^2/(Np)')
axes[1].scatter(p_vals, numerical_vals, color='red', s=60, zorder=5,
                label='Simulation (mean of 5 graphs)')
axes[1].set_xlabel('Rewiring probability p')
axes[1].set_ylabel('FSRI Omega')
axes[1].set_title(f'Exact formula Omega(K=6,p)\nclosed form against simulation (N=100)\nr={corr_formula:.3f}')
axes[1].legend(fontsize=8)
axes[1].set_yscale('log')
axes[1].grid(alpha=0.3)

# Plot 3: corrected relation
axes[2].scatter(ratio_vals, omega_p_vals, alpha=0.5, c='steelblue', s=40,
                label=f'r={corr_corrected:.3f}')
z = np.polyfit(ratio_vals, omega_p_vals, 1)
p_fit = np.poly1d(z)
x_line = np.linspace(min(ratio_vals), max(ratio_vals), 100)
axes[2].plot(x_line, p_fit(x_line), 'r--', lw=2)
axes[2].set_xlabel('(S_env/E_ext)^2 (1-p)')
axes[2].set_ylabel('Omega p')
axes[2].set_title(f'Corrected relation\nOmega p ~ (S_env/E_ext)^2 (1-p)\nr={corr_corrected:.3f}')
axes[2].legend(fontsize=10)
axes[2].grid(alpha=0.3)

plt.suptitle('FSRI: mathematical validation\n'
             'ORC and Tr(A^3), the exact WS formula, the corrected relation',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('fsri_validation_3.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("CONCLUSIONS: MATHEMATICAL VALIDATION")
print("="*65)
print(f"1. The ORC kappa(x,y) tracks the number of common neighbours,")
print(f"   which is what relates the curvature to Tr(A^3)")
print(f"2. Exact WS formula: Omega = (3K-6)/(4K-2)(1-p)^2/(N p)")
print(f"   Correlation of closed form with simulation: {corr_formula:.4f}")
print(f"   Validated on WS with K=6, N=100, p in [0.05, 0.9]")
print(f"3. Omega is INVERSELY related to (S_env/E_ext)^2, not proportional to it")
print(f"   r={corr_direct:.4f} for the direct form, and the exact identity")
print(f"   puts the ratio in the denominator with relative error {max_rel:.1e}")
print(f"   An earlier formulation of this framework proposed a proportionality,")
print(f"   which holds only as p tends to zero")
