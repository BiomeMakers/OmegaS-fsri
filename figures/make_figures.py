"""
Conceptual figures for the FSRI preprint.

All three are computed, not drawn by hand: the graphs are constructed in code
and every quantity shown in a label is evaluated from the adjacency matrix.
Figure 2 in particular searches for the cospectral pair rather than hard-coding
it, so the counterexample in the text is verifiable by running this file.
"""
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "figure.dpi": 200})
BLUE, RED, GREY, GREEN = "#1f4e79", "#c0504d", "#8c8c8c", "#4f7942"


def circle_layout(n, r=1.0, rot=np.pi / 2):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + rot
    return np.c_[r * np.cos(a), r * np.sin(a)]


def draw(ax, A, pos, node_vals=None, node_col=None, title="", vmax=None,
         edge_col=None, node_size=430):
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] != 0:
                c = edge_col(A[i, j]) if edge_col else "0.55"
                lw = 1.1 if edge_col is None else 1.6
                ax.plot(*zip(pos[i], pos[j]), color=c, lw=lw, zorder=1)
    if node_vals is not None and vmax:
        cols = [plt.cm.Blues(0.25 + 0.7 * v / vmax) for v in node_vals]
    else:
        cols = node_col if node_col else [BLUE] * n
    ax.scatter(pos[:, 0], pos[:, 1], s=node_size, c=cols, zorder=3,
               edgecolors="white", linewidths=1.4)
    if node_vals is not None:
        for p, v in zip(pos, node_vals):
            ax.text(p[0], p[1], f"{v:g}", ha="center", va="center",
                    fontsize=8, color="white", zorder=4, fontweight="bold")
    ax.set_title(title, fontsize=9.5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.45)


def tr3(A):
    return float(np.trace(A @ A @ A))


def diag3(A):
    return np.diag(A @ A @ A).astype(int)


# ---------------------------------------------------------------- Figure 1
# What the index measures: a distributed network versus a structural monopoly,
# at matched edge count.
n = 7
Amon = np.zeros((n, n))                      # hub plus a tight core
for j in range(1, n):
    Amon[0, j] = Amon[j, 0] = 1
for j in (1, 2, 3):
    for k_ in (1, 2, 3):
        if j != k_:
            Amon[j, k_] = 1
Adis = np.zeros((n, n))                      # ring plus chords, same edges
ring = [(i, (i + 1) % n) for i in range(n)]
for i, j in ring:
    Adis[i, j] = Adis[j, i] = 1
for i, j in [(0, 2), (1, 4), (3, 5), (2, 6), (4, 6), (0, 3)]:
    Adis[i, j] = Adis[j, i] = 1

pos = circle_layout(n)
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.9))
for a, A, name in [(ax[0], Adis, "distributed"), (ax[1], Amon, "monopolised")]:
    deg = A.sum(1)
    t = tr3(A) / 6
    ttl = (f"{name}\n"
           f"$\\mathrm{{Tr}}(A^3)/6$ = {t:.0f} triangles, "
           f"Var(deg) = {np.var(deg):.2f}")
    draw(a, A, pos, node_vals=diag3(A) // 2, vmax=max(diag3(A) // 2) or 1,
         title=ttl)
fig.suptitle("Node labels: triangles incident on that node, "
             r"$(A^3)_{ii}/2$", fontsize=8.5, y=0.04, color="0.35")
fig.tight_layout()
fig.savefig("figure1_monopoly.png", bbox_inches="tight")
fig.savefig("figure1_monopoly.pdf", bbox_inches="tight")
print("figure1_monopoly written")

# ---------------------------------------------------------------- Figure 2
# Cospectral counterexample: identical spectrum and identical Tr(A^3),
# different diag(A^3). Found by exhaustive search, not hard-coded.
def find_cospectral(nv=6):
    pairs = list(itertools.combinations(range(nv), 2))
    seen = {}
    for mask in range(1 << len(pairs)):
        if bin(mask).count("1") < 6:
            continue
        M = np.zeros((nv, nv))
        for b, (i, j) in enumerate(pairs):
            if mask >> b & 1:
                M[i, j] = M[j, i] = 1
        key = tuple(np.round(np.linalg.eigvalsh(M), 6))
        d = tuple(sorted(diag3(M)))
        if key in seen and seen[key][1] != d:
            return seen[key][0], M
        seen.setdefault(key, (M.copy(), d))
    raise RuntimeError("no cospectral pair found")


G1, G2 = find_cospectral()
spec = np.round(np.linalg.eigvalsh(G1), 4)
assert np.allclose(spec, np.round(np.linalg.eigvalsh(G2), 4))
assert np.isclose(tr3(G1), tr3(G2))

pos6 = circle_layout(6)
fig2, ax2 = plt.subplots(1, 2, figsize=(8.4, 4.2))
plt.rcParams["axes.titlesize"] = 8.5
# Format by hand: under numpy 2.x, tuple() over numpy scalars prints the full
# repr (np.int64(4), np.float64(-1.9032)), which ended up inside the figure.
def _fmt_ints(v):
    return "(" + ", ".join(f"{int(round(x))}" for x in v) + ")"


def _fmt_floats(v):
    return "(" + ", ".join(f"{float(x):g}" for x in v) + ")"


for a, G, lab in [(ax2[0], G1, "$G_1$"), (ax2[1], G2, "$G_2$")]:
    dv = diag3(G)
    draw(a, G, pos6, node_vals=dv, vmax=max(dv) or 1,
         title=f"{lab}\n$\\mathrm{{diag}}(A^3)$ = {_fmt_ints(dv)}")
txt = (f"identical spectrum  {_fmt_floats(spec)}"
       f"        identical  $\\mathrm{{Tr}}(A^3)$ = {tr3(G1):.0f}")
fig2.suptitle(txt, fontsize=8.5, y=0.045, color="0.35")
fig2.tight_layout()
fig2.savefig("figure2_cospectral.png", bbox_inches="tight")
fig2.savefig("figure2_cospectral.pdf", bbox_inches="tight")
print("figure2_cospectral written")

# ---------------------------------------------------------------- Figure 3
# Signed extension: balanced versus frustrated triangles.
Bal = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], float)
Fru = np.array([[0, 1, 1], [1, 0, -1], [1, -1, 0]], float)
Abs = np.abs(Fru)
pos3 = circle_layout(3, r=0.95)


def ecol(w):
    return BLUE if w > 0 else RED


fig3, ax3 = plt.subplots(1, 3, figsize=(8.6, 3.2))
for a, A, lab in [(ax3[0], Bal, "balanced"),
                  (ax3[1], Fru, "frustrated"),
                  (ax3[2], Abs, r"either, seen through $|A|$")]:
    signed = not np.allclose(A, np.abs(A)) or lab == "balanced"
    draw(a, A, pos3, node_col=[GREY] * 3, node_size=330,
         edge_col=ecol if lab != r"either, seen through $|A|$" else (lambda w: "0.5"),
         title=f"{lab}\n" + r"$\mathrm{Tr}(A^3)$ = " + f"{tr3(A):+.0f}")
fig3.suptitle("Blue: co-occurrence (+).  Red: co-exclusion (-).  "
              r"Taking $|A|$ discards the distinction.",
              fontsize=8.5, y=0.02, color="0.35")
fig3.tight_layout()
fig3.savefig("figure3_signed.png", bbox_inches="tight")
fig3.savefig("figure3_signed.pdf", bbox_inches="tight")
print("figure3_signed written")
