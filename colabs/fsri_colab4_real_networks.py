# =============================================================================
# FSRI Validation - Notebook 4: the ORC bounds on real networks
# Author: Alberto Acedo (acedo@biomemakers.com)
# =============================================================================
"""
Checks the ORC bounds empirically, using the FAST Jost-Liu approximation:

  Tr(A^3) / (2 d_max) <= Sum k+(x,y) <= Tr(A^3) / (2 d_min)

Only the upper bound is proven in the paper. Note also that this notebook uses
the approximation, which underestimates the curvature on heterogeneous graphs.
Notebook 4b computes the exact Wasserstein-1 curvature and is the one that
produces Table 3 of the paper; prefer it when the numbers matter.

Networks (all bundled with NetworkX, no download needed):
  1. Karate Club (Zachary 1977), 34 nodes
  2. Florentine Families, the historical marriage and credit network
  3. Les Miserables, co-appearance network, 77 nodes
  4. Davis Southern Women, bipartite social network, projected
  5. Grid 8x8, regular control with d_max/d_min = 1
  6. Erdos-Renyi (n=80, p=0.12)
  7. Barabasi-Albert (n=80, m=3)

The two bounds are reported SEPARATELY, because only the upper one is proven.
The conjectured lower bound fails on the sparse random graphs, and the paper
claims only the upper bound for exactly that reason.

CPU only, about three minutes. No extra installation required.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. FUNCIONES BASE
# ---------------------------------------------------------------------------

def orc_jost_liu(G, x, y):
    common = len(set(G.neighbors(x)) & set(G.neighbors(y)) - {x, y})
    dx, dy = G.degree(x), G.degree(y)
    if dx == 0 or dy == 0:
        return 0.0
    return common * (1/dx + 1/dy) - 1 + 1/dx + 1/dy

def sum_kappa_plus(G):
    return sum(max(orc_jost_liu(G, x, y), 0.0) for x, y in G.edges())

def tr_a3(G):
    return 6 * (sum(nx.triangles(G).values()) // 3)

def validate(G, name):
    degrees = [d for _, d in G.degree()]
    d_max = max(degrees)
    d_min = max(min(degrees), 1)
    tra3  = tr_a3(G)
    skp   = sum_kappa_plus(G)
    lb    = tra3 / (2 * d_max)
    ub    = tra3 / (2 * d_min)
    ratio = skp / max(ub, 1e-10)
    # The two bounds are checked SEPARATELY. Only the upper one is proven.
    valid_ub = (skp <= ub) if ub > 0 else (tra3 == 0)
    valid_lb = (skp >= lb) if ub > 0 else (tra3 == 0)
    return {"name": name, "n": G.number_of_nodes(), "m": G.number_of_edges(),
            "d_max": d_max, "d_min": d_min, "het": d_max/d_min,
            "tra3": tra3, "lb": lb, "skp": skp, "ub": ub,
            "ratio": ratio, "valid_ub": valid_ub, "valid_lb": valid_lb}

# ---------------------------------------------------------------------------
# 2. LOAD THE NETWORKS
# ---------------------------------------------------------------------------

print("Loading networks...")
nets = {}
nets["Karate Club\n(Zachary 1977)"]          = nx.karate_club_graph()
nets["Florentine Families\n(Padgett-Ansell)"] = nx.florentine_families_graph()
nets["Les Miserables\n(co-appearance)"]       = nx.les_miserables_graph()

G_dsw = nx.davis_southern_women_graph()
women = {n for n, d in G_dsw.nodes(data=True) if d.get("bipartite", 0) == 0}
G_women = nx.bipartite.projected_graph(G_dsw, women)
nets["Davis Southern Women\n(projected)"]     = G_women

G_grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(8, 8))
nets["Grid 8x8\n(regular control)"]   = G_grid
nets["Erdos-Renyi\n(n=80, p=0.12)"]   = nx.erdos_renyi_graph(80, 0.12, seed=42)
nets["Barabasi-Albert\n(n=80, m=3)"]  = nx.barabasi_albert_graph(80, 3, seed=42)

# ---------------------------------------------------------------------------
# 3. CHECK THE BOUNDS
# ---------------------------------------------------------------------------

print("="*90)
print("THE ORC BOUNDS ON REAL NETWORKS (Jost-Liu approximation)")
print("Proven upper bound:      Sum k+ <= Tr(A3)/(2*d_min)")
print("Conjectured lower bound: Sum k+ >= Tr(A3)/(2*d_max)")
print("="*90)

results = []
print(f"\n{'Network':<30} {'N':>5} {'M':>6} {'het':>6} {'LB':>8} "
      f"{'Sum k+':>8} {'UB':>8} {'k+/UB':>7} {'UB ok':>7} {'LB ok':>7}")
print("-"*98)

for name, G in nets.items():
    if G.number_of_edges() == 0:
        continue
    r = validate(G, name)
    results.append(r)
    short = name.split('\n')[0]
    mark_ub = "yes" if r["valid_ub"] else "NO"
    mark_lb = "yes" if r["valid_lb"] else "NO"
    print(f"{short:<30} {r['n']:>5} {r['m']:>6} {r['het']:>6.2f} "
          f"{r['lb']:>8.2f} {r['skp']:>8.2f} {r['ub']:>8.2f} "
          f"{r['ratio']:>7.3f} {mark_ub:>7} {mark_lb:>7}")

n_ub = sum(1 for r in results if r["valid_ub"])
n_lb = sum(1 for r in results if r["valid_lb"])
print(f"\nProven upper bound satisfied:      {n_ub}/{len(results)} networks")
print(f"Conjectured lower bound satisfied: {n_lb}/{len(results)} networks")

# ---------------------------------------------------------------------------
# 4. TIGHTNESS
# ---------------------------------------------------------------------------

print("\n" + "="*80)
print("SLACK IN THE PROVEN BOUND: Sum k+ / UB, which must lie in (0, 1]")
print("="*80)

valid_r = [r for r in results if r["valid_ub"]]
for r in valid_r:
    print(f"  {r['name'].split(chr(10))[0]:<30}: "
          f"k+/UB={r['ratio']:.3f}  het={r['het']:.2f}")

if len(valid_r) >= 3:
    hets   = [r["het"]   for r in valid_r]
    ratios = [r["ratio"] for r in valid_r]
    corr_h, pval_h = pearsonr(hets, ratios)
    print(f"\nCorrelation of heterogeneity with slack: r={corr_h:.4f} (p={pval_h:.4f})")
    print("Note: the paper finds that d_min, not clustering, governs the slack.")

# ---------------------------------------------------------------------------
# 5. Tr(A3) AGAINST Sum k+
# ---------------------------------------------------------------------------

print("\n" + "="*80)
print("log(Tr(A3)) AGAINST log(Sum k+)")
print("="*80)

pairs = [(r["tra3"], r["skp"]) for r in results if r["tra3"] > 0 and r["skp"] > 0]
corr_p = 0
if len(pairs) >= 3:
    t_vals, k_vals = zip(*pairs)
    corr_p, pval_p = pearsonr(np.log(t_vals), np.log(k_vals))
    slope = np.polyfit(np.log(t_vals), np.log(k_vals), 1)[0]
    print(f"r={corr_p:.4f} (p={pval_p:.4f}), log-log slope={slope:.3f}")
    print(f"-> {'strong association' if corr_p > 0.9 else 'positive but weak association'}")

# ---------------------------------------------------------------------------
# 6. FIGURES
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: the two bounds, by network
names_s = [r["name"].split('\n')[0] for r in valid_r]
lbs  = [r["lb"]  for r in valid_r]
skps = [r["skp"] for r in valid_r]
ubs  = [r["ub"]  for r in valid_r]
xs   = np.arange(len(valid_r))

axes[0].fill_between(xs, lbs, ubs, alpha=0.25, color='green', label='[LB, UB]')
axes[0].plot(xs, lbs, 'g--', lw=1.5, label='LB (conjectured)')
axes[0].plot(xs, ubs, 'g:',  lw=1.5, label='UB (proven)')
axes[0].scatter(xs, skps, color='red', s=80, zorder=5, label='Sum k+ observed')
axes[0].set_xticks(xs)
axes[0].set_xticklabels(names_s, rotation=35, ha='right', fontsize=7)
axes[0].set_ylabel('Value')
axes[0].set_title('ORC bounds\non real networks')
axes[0].legend(fontsize=7)
axes[0].set_yscale('symlog', linthresh=0.1)
axes[0].grid(alpha=0.3)

# Plot 2: log-log association
if len(pairs) >= 3:
    axes[1].scatter(np.log(t_vals), np.log(k_vals), c='steelblue', s=80, zorder=5)
    for r in results:
        if r["tra3"] > 0 and r["skp"] > 0:
            axes[1].annotate(r["name"].split('\n')[0],
                             (np.log(r["tra3"]), np.log(r["skp"])),
                             textcoords="offset points", xytext=(4, 3), fontsize=7)
    z    = np.polyfit(np.log(t_vals), np.log(k_vals), 1)
    xfit = np.linspace(min(np.log(t_vals)), max(np.log(t_vals)), 100)
    axes[1].plot(xfit, np.poly1d(z)(xfit), 'r--', lw=2,
                 label=f'slope={z[0]:.3f}, r={corr_p:.3f}')
    axes[1].set_xlabel('log Tr(A3)')
    axes[1].set_ylabel('log Sum k+')
    axes[1].set_title('Tr(A3) against\nSum k+')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

# Plot 3: slack against heterogeneity
hets_all   = [r["het"]   for r in valid_r]
ratios_all = [r["ratio"] for r in valid_r]
axes[2].scatter(hets_all, ratios_all,
                c=['green' if r["valid_ub"] else 'red' for r in valid_r],
                s=80, zorder=5)
for r in valid_r:
    axes[2].annotate(r["name"].split('\n')[0], (r["het"], r["ratio"]),
                     textcoords="offset points", xytext=(4, 3), fontsize=7)
axes[2].axhline(y=1, color='r', linestyle='--', lw=2,
                label='UB (points must stay below)')
axes[2].set_ylim(0, 1.1)
axes[2].set_xlabel('d_max / d_min (heterogeneity)')
axes[2].set_ylabel('Sum k+ / UB')
axes[2].set_title('Slack in the proven bound\nagainst degree heterogeneity')
axes[2].legend(fontsize=7)
axes[2].grid(alpha=0.3)

plt.suptitle('The ORC bounds on real networks (Jost-Liu approximation)\n'
             'proven upper bound Sum k+ <= Tr(A3)/(2*d_min); '
             'the lower bound is conjectured only',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('fsri_validation_4_real_networks.png', dpi=150, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------------------------
# 7. SUMMARY
# ---------------------------------------------------------------------------

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"1. Proven upper bound:      {n_ub}/{len(results)} networks")
print(f"2. Conjectured lower bound: {n_lb}/{len(results)} networks")
print(f"   The lower bound is not claimed in the paper. Notebook 4b repeats")
print(f"   this with exact optimal transport and it still fails there, so the")
print(f"   failure is a property of the bound and not of the approximation.")
if len(pairs) >= 3:
    print(f"3. log Tr(A3) against log Sum k+: r={corr_p:.4f}")
if len(valid_r) >= 3:
    print(f"4. Heterogeneity against slack: r={corr_h:.4f}")
fam_r = next((r for r in results if "Florentine" in r["name"]), None)
if fam_r:
    print(f"5. Florentine Families (Padgett and Ansell 1993):")
    print(f"   Tr(A3)={fam_r['tra3']}, Sum k+={fam_r['skp']:.2f}, "
          f"LB={fam_r['lb']:.2f}, UB={fam_r['ub']:.2f}")
print(f"\nNote: these values use the Jost-Liu approximation and differ from")
print(f"Table 3 of the paper, which is produced by notebook 4b with exact")
print(f"Wasserstein-1 transport.")
