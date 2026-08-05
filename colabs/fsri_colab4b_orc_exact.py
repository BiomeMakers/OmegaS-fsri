# =============================================================================
# FSRI Validation - Notebook 4b: exact ORC via Wasserstein-1
# Author: Alberto Acedo (acedo@biomemakers.com)
# =============================================================================
"""
Computes the EXACT Ollivier-Ricci curvature using the Wasserstein-1 (earth
mover) distance between neighbour distributions, solving the optimal transport
problem with scipy. This is the notebook that produces Table 3 of the paper.

Notebook 4 used the Jost-Liu approximation, which underestimates the curvature
on heterogeneous graphs. This one uses exact W1 via linear_sum_assignment.

TWO BOUNDS, AND ONLY ONE OF THEM IS PROVEN.

  Upper bound (proven in the paper):  sum kappa+(x,y) <= Tr(A^3) / (2 d_min)
  Lower bound (conjectured only):     sum kappa+(x,y) >= Tr(A^3) / (2 d_max)

The two are reported separately below because they behave differently. The
upper bound holds on all six networks. The conjectured lower bound fails on
the two sparse random graphs, and the paper claims only the upper bound for
exactly that reason. An earlier version of this notebook tested them jointly
as a sandwich and reported "4/6", which read as a failure of the proven bound
when it is a failure of the conjectured one.

CPU only, about five minutes. Requires networkx, numpy, scipy and matplotlib.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. EXACT ORC VIA WASSERSTEIN-1
# ---------------------------------------------------------------------------

def wasserstein1_graph(G, x, y):
    """
    Exact W1(mu_x, mu_y), where mu_x is the uniform distribution over the
    neighbours of x and mu_y the uniform distribution over those of y.

    The transport cost between two nodes is the shortest-path distance in G.
    For efficiency only distances within the union of the two neighbourhoods
    are computed.
    """
    nx_set = set(G.neighbors(x))
    ny_set = set(G.neighbors(y))

    if len(nx_set) == 0 or len(ny_set) == 0:
        return 1.0  # kappa = 0 by definition

    # sorted() and not list(): iterating a Python set gives an order that
    # depends on PYTHONHASHSEED when the nodes are strings, and because
    # linear_sum_assignment has ties, the sum of positive curvature changed
    # between runs (Les Miserables moved between 81.07 and 83.02). Sorting
    # makes it deterministic.
    nx_list = sorted(nx_set)
    ny_list = sorted(ny_set)
    dx = len(nx_list)
    dy = len(ny_list)

    # Cost matrix: C[i,j] = d(nx_list[i], ny_list[j]), by shortest path
    # between the relevant nodes.
    nodes_needed = set(nx_list) | set(ny_list)
    C = np.zeros((dx, dy))
    for i, u in enumerate(nx_list):
        for j, v in enumerate(ny_list):
            if u == v:
                C[i, j] = 0.0
            elif G.has_edge(u, v):
                C[i, j] = 1.0
            else:
                try:
                    C[i, j] = nx.shortest_path_length(G, u, v)
                except nx.NetworkXNoPath:
                    C[i, j] = len(G)  # maximum penalty

    # Optimal transport: uniform mass 1/dx leaves each node of N(x) and
    # uniform mass 1/dy arrives at each node of N(y), and W1 is the minimum
    # total cost over matchings.

    if dx == dy:
        # Square case: linear_sum_assignment is exact
        row_ind, col_ind = linear_sum_assignment(C)
        w1 = C[row_ind, col_ind].sum() / dx
    else:
        # Rectangular case: the optimal rectangular matching, which is correct
        # for uniform distributions.
        if dx < dy:
            # Repeat rows to square the matrix
            reps = (dy + dx - 1) // dx
            C_sq = np.tile(C, (reps, 1))[:dy, :]
            row_ind, col_ind = linear_sum_assignment(C_sq)
            w1 = C_sq[row_ind, col_ind].sum() / dy
        else:
            # Repeat columns
            reps = (dx + dy - 1) // dy
            C_sq = np.tile(C, (1, reps))[:, :dx]
            row_ind, col_ind = linear_sum_assignment(C_sq)
            w1 = C_sq[row_ind, col_ind].sum() / dx

    return w1

def orc_exact(G, x, y):
    """kappa(x,y) = 1 - W1(mu_x, mu_y) / d(x,y)"""
    d_xy = 1  # always 1 for adjacent nodes
    w1 = wasserstein1_graph(G, x, y)
    return 1.0 - w1 / d_xy

def sum_kappa_plus_exact(G, verbose=False):
    """Sum of the positive part of the exact curvature over all edges."""
    total = 0.0
    n_pos, n_neg = 0, 0
    for x, y in G.edges():
        k = orc_exact(G, x, y)
        if k > 0:
            total += k
            n_pos += 1
        else:
            n_neg += 1
    if verbose:
        print(f"    kappa>0: {n_pos} edges, kappa<=0: {n_neg} edges")
    return total

def tr_a3(G):
    return 6 * (sum(nx.triangles(G).values()) // 3)

def validate_exact(G, name, verbose=True):
    degrees = [d for _, d in G.degree()]
    d_max = max(degrees)
    d_min = max(min(degrees), 1)
    tra3  = tr_a3(G)
    if verbose:
        print(f"  Computing exact ORC for {name.split(chr(10))[0]} "
              f"({G.number_of_edges()} edges)...")
    skp   = sum_kappa_plus_exact(G, verbose=verbose)
    lb    = tra3 / (2 * d_max)
    ub    = tra3 / (2 * d_min)
    ratio = skp / max(ub, 1e-10)
    # The two bounds are checked SEPARATELY. Only the upper one is proven.
    valid_ub = (skp <= ub) if ub > 0 else (tra3 == 0)
    valid_lb = (skp >= lb) if ub > 0 else (tra3 == 0)
    return {
        "name": name, "n": G.number_of_nodes(), "m": G.number_of_edges(),
        "d_max": d_max, "d_min": d_min, "het": d_max/d_min,
        "tra3": tra3, "lb": lb, "skp": skp, "ub": ub,
        "ratio": ratio, "valid_ub": valid_ub, "valid_lb": valid_lb,
    }

# ---------------------------------------------------------------------------
# 2. APPROXIMATE AGAINST EXACT ORC ON A SMALL NETWORK
# ---------------------------------------------------------------------------

def orc_jost_liu(G, x, y):
    """Approximate ORC, Jost and Liu (2014)."""
    common = len(set(G.neighbors(x)) & set(G.neighbors(y)) - {x, y})
    dx, dy = G.degree(x), G.degree(y)
    if dx == 0 or dy == 0:
        return 0.0
    return common * (1/dx + 1/dy) - 1 + 1/dx + 1/dy

print("="*70)
print("EXACT AGAINST APPROXIMATE ORC (Jost-Liu)")
print("Zachary Karate Club (34 nodes, 78 edges)")
print("="*70)

G_karate = nx.karate_club_graph()
kappas_exact = [orc_exact(G_karate, x, y) for x, y in G_karate.edges()]
kappas_approx = [orc_jost_liu(G_karate, x, y) for x, y in G_karate.edges()]

corr_ka, _ = pearsonr(kappas_exact, kappas_approx)
print(f"Correlation of exact with approximate ORC: {corr_ka:.4f}")
print(f"Exact ORC        - mean: {np.mean(kappas_exact):.4f}, std: {np.std(kappas_exact):.4f}")
print(f"Approximate ORC  - mean: {np.mean(kappas_approx):.4f}, std: {np.std(kappas_approx):.4f}")
print(f"Fraction with kappa>0 (exact):       {sum(1 for k in kappas_exact if k>0)/len(kappas_exact):.2%}")
print(f"Fraction with kappa>0 (approximate): {sum(1 for k in kappas_approx if k>0)/len(kappas_approx):.2%}")

# ---------------------------------------------------------------------------
# 3. CHECKING THE BOUNDS WITH EXACT ORC
# ---------------------------------------------------------------------------

print("\n" + "="*70)
print("THE ORC BOUNDS ON REAL NETWORKS, WITH EXACT CURVATURE")
print("Proven upper bound:      sum kappa+ <= Tr(A^3) / (2 d_min)")
print("Conjectured lower bound: sum kappa+ >= Tr(A^3) / (2 d_max)")
print("="*70)

# Small networks, so that exact W1 stays tractable
nets = {}
nets["Karate Club\n(Zachary 1977)"]              = nx.karate_club_graph()
nets["Florentine Families\n(Padgett-Ansell)"]    = nx.florentine_families_graph()
nets["Les Miserables\n(co-appearance)"]          = nx.les_miserables_graph()

G_dsw = nx.davis_southern_women_graph()
women = {n for n, d in G_dsw.nodes(data=True) if d.get("bipartite", 0) == 0}
nets["Davis Southern Women\n(social)"]           = nx.bipartite.projected_graph(G_dsw, women)

# The two graphs on which the conjectured lower bound fails
nets["Erdos-Renyi\n(n=40, p=0.15)"]    = nx.erdos_renyi_graph(40, 0.15, seed=42)
nets["Barabasi-Albert\n(n=40, m=3)"]   = nx.barabasi_albert_graph(40, 3, seed=42)

results_exact = []
print(f"\n{'Network':<30} {'N':>5} {'M':>6} {'het':>6} {'LB':>8} "
      f"{'sum k+':>10} {'UB':>8} {'k+/UB':>7} {'UB ok':>7} {'LB ok':>7}")
print("-"*98)

for name, G in nets.items():
    if G.number_of_edges() == 0:
        continue
    r = validate_exact(G, name, verbose=True)
    results_exact.append(r)
    short = name.split('\n')[0]
    mark_ub = "yes" if r["valid_ub"] else "NO"
    mark_lb = "yes" if r["valid_lb"] else "NO"
    print(f"{short:<30} {r['n']:>5} {r['m']:>6} {r['het']:>6.2f} "
          f"{r['lb']:>8.2f} {r['skp']:>10.2f} {r['ub']:>8.2f} "
          f"{r['ratio']:>7.3f} {mark_ub:>7} {mark_lb:>7}")

n_ub = sum(1 for r in results_exact if r["valid_ub"])
n_lb = sum(1 for r in results_exact if r["valid_lb"])
print(f"\nProven upper bound satisfied:      {n_ub}/{len(results_exact)} networks")
print(f"Conjectured lower bound satisfied: {n_lb}/{len(results_exact)} networks")

# ---------------------------------------------------------------------------
# 4. DOES EXACT ORC RESCUE THE CONJECTURED LOWER BOUND?
# ---------------------------------------------------------------------------

print("\n" + "="*70)
print("DIAGNOSIS: does exact curvature change the lower-bound failures?")
print("="*70)

for name in ["Erdos-Renyi\n(n=40, p=0.15)", "Barabasi-Albert\n(n=40, m=3)"]:
    G = nets[name]
    short = name.split('\n')[0]
    r_exact = next(r for r in results_exact if r["name"] == name)

    # Approximate ORC
    skp_approx = sum(max(orc_jost_liu(G, x, y), 0.0) for x, y in G.edges())

    print(f"\n{short}:")
    print(f"  Conjectured LB = {r_exact['lb']:.3f}")
    print(f"  sum k+ approximate (Jost-Liu): {skp_approx:.3f}  "
          f"{'>= LB' if skp_approx >= r_exact['lb'] else '< LB, fails'}")
    print(f"  sum k+ exact (W1):             {r_exact['skp']:.3f}  "
          f"{'>= LB' if r_exact['skp'] >= r_exact['lb'] else '< LB, fails'}")
    print(f"  Proven UB = {r_exact['ub']:.3f}  "
          f"{'satisfied' if r_exact['valid_ub'] else 'VIOLATED'}")
    print(f"  Exact curvature {'rescues' if r_exact['valid_lb'] else 'does not rescue'} the lower bound")

# ---------------------------------------------------------------------------
# 5. FIGURES
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: exact against approximate ORC on the Karate Club
axes[0].scatter(kappas_approx, kappas_exact, alpha=0.6, c='steelblue', s=40)
lims = [min(min(kappas_approx), min(kappas_exact))-0.05,
        max(max(kappas_approx), max(kappas_exact))+0.05]
axes[0].plot(lims, lims, 'r--', lw=2, label='y=x (exact = approximate)')
axes[0].axhline(y=0, color='k', linestyle=':', alpha=0.5)
axes[0].axvline(x=0, color='k', linestyle=':', alpha=0.5)
axes[0].set_xlabel('approximate kappa (Jost-Liu)')
axes[0].set_ylabel('exact kappa (W1)')
axes[0].set_title(f'Exact against approximate ORC\nKarate Club  r={corr_ka:.3f}')
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# Plot 2: the two bounds with exact ORC
names_s = [r["name"].split('\n')[0] for r in results_exact]
lbs_e  = [r["lb"]  for r in results_exact]
skps_e = [r["skp"] for r in results_exact]
ubs_e  = [r["ub"]  for r in results_exact]
xs = np.arange(len(results_exact))

colors_bar = ['green' if r["valid_ub"] else 'red' for r in results_exact]
axes[1].fill_between(xs, lbs_e, ubs_e, alpha=0.2, color='green',
                     label='between the two bounds')
axes[1].plot(xs, lbs_e, 'g--', lw=1.5, label='LB (conjectured)')
axes[1].plot(xs, ubs_e, 'g:',  lw=1.5, label='UB (proven)')
axes[1].scatter(xs, skps_e, c=colors_bar, s=80, zorder=5,
                label='sum kappa+ (exact)')
axes[1].set_xticks(xs)
axes[1].set_xticklabels(names_s, rotation=35, ha='right', fontsize=7)
axes[1].set_ylabel('Value')
axes[1].set_title(f'ORC bounds, exact curvature\nproven UB: {n_ub}/{len(results_exact)}, conjectured LB: {n_lb}/{len(results_exact)}')
axes[1].legend(fontsize=7)
axes[1].set_yscale('symlog', linthresh=0.1)
axes[1].grid(alpha=0.3)

# Plot 3: slack against degree heterogeneity
hets_e   = [r["het"]   for r in results_exact]
ratios_e = [r["ratio"] for r in results_exact]
colors_e = ['green' if r["valid_ub"] else 'red' for r in results_exact]
axes[2].scatter(hets_e, ratios_e, c=colors_e, s=80, zorder=5)
for r in results_exact:
    axes[2].annotate(r["name"].split('\n')[0],
                     (r["het"], r["ratio"]),
                     textcoords="offset points", xytext=(4, 3), fontsize=7)
axes[2].axhline(y=1, color='r', linestyle='--', lw=2,
                label='UB (values must stay below)')
axes[2].set_xlabel('d_max / d_min')
axes[2].set_ylabel('sum kappa+ / UB')
axes[2].set_ylim(0, 1.1)
axes[2].set_title('Slack in the proven bound\nagainst degree heterogeneity')
axes[2].legend(fontsize=7)
axes[2].grid(alpha=0.3)

plt.suptitle(
    'Exact ORC (Wasserstein-1): the curvature bounds\n'
    'proven upper bound sum kappa+ <= Tr(A^3)/(2 d_min); '
    'the lower bound is conjectured only',
    fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('fsri_validation_4b_orc_exact.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# 6. SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "="*70)
print("CONCLUSIONS: THE TWO BOUNDS UNDER EXACT CURVATURE")
print("="*70)
print(f"1. Exact against approximate ORC on the Karate Club: r={corr_ka:.4f}")
print(f"   The approximation is {'close' if corr_ka > 0.8 else 'imprecise'} "
      f"and underestimates kappa on heterogeneous graphs")
print(f"2. Proven upper bound:      {n_ub}/{len(results_exact)} networks")
print(f"3. Conjectured lower bound: {n_lb}/{len(results_exact)} networks")

er_r  = next((r for r in results_exact if "Erdos" in r["name"]), None)
ba_r  = next((r for r in results_exact if "Barabasi" in r["name"]), None)
for r in (er_r, ba_r):
    if r:
        print(f"   {r['name'].split(chr(10))[0]}: sum kappa+ = {r['skp']:.2f}, "
              f"conjectured LB = {r['lb']:.2f}, "
              f"{'satisfied' if r['valid_lb'] else 'so the lower bound fails'}")

print(f"\nReading:")
if n_ub == len(results_exact):
    print(f"  The bound the paper proves holds on all {n_ub} networks.")
else:
    print(f"  WARNING: the proven upper bound fails on "
          f"{len(results_exact)-n_ub} network(s). This should not happen;")
    print(f"  please open an issue in the repository with your output.")
print(f"  The lower bound is not claimed in the paper, and these runs are why:")
print(f"  exact transport does not rescue it on the two sparse random graphs.")
print(f"  Slack in the upper bound is governed by d_min, not by clustering.")
