"""
FSRI Colab 5 : Spectral identity and the locus of non-spectral information.
Verifies numerically the propositions of Section 2.4:
  (1) Tr(A^3) = sum_i lambda_i^3          -> the global functional IS spectral
  (2) (A^3)_ii = sum_k lambda_k^3 v_ki^2  -> node decomposition needs eigenvectors
  (3) explicit COSPECTRAL pair with equal Tr(A^3) but different diag(A^3)
      -> diag(A^3) is NOT determined by the spectrum
  (4) signed Tr(A^3) = 6*(balanced - unbalanced triangles)  [Heider balance]
"""
import numpy as np, itertools

rng = np.random.default_rng(0)
A = rng.standard_normal((8, 8)); A = (A + A.T) / 2; np.fill_diagonal(A, 0)
lam, V = np.linalg.eigh(A)
print("(1) Tr(A^3) == sum lambda^3      :", np.isclose(np.trace(A@A@A), (lam**3).sum()))
print("(2) (A^3)_ii == sum_k l_k^3 v_ki^2:",
      np.allclose(np.diag(A@A@A), np.einsum('k,ik->i', lam**3, V**2)))

# (3) exhaustive search for a cospectral pair on 6 vertices with different diag(A^3)
n = 6; pairs = list(itertools.combinations(range(n), 2)); seen = {}; found = None
for mask in range(1 << len(pairs)):
    if bin(mask).count('1') < 6: continue
    M = np.zeros((n, n))
    for b, (i, j) in enumerate(pairs):
        if mask >> b & 1: M[i, j] = M[j, i] = 1
    key = tuple(np.round(np.linalg.eigvalsh(M), 6))
    d = tuple(sorted(np.diag(M@M@M).astype(int)))
    if key in seen and seen[key][1] != d:
        found = (seen[key], (M.copy(), d)); break
    seen.setdefault(key, (M.copy(), d))
(M1, d1), (M2, d2) = found
print("\n(3) cospectral pair found")
print("    common spectrum :", np.round(np.linalg.eigvalsh(M1), 4))
print("    Tr(A^3) both    :", round(np.trace(M1@M1@M1), 4), "vs", round(np.trace(M2@M2@M2), 4))
fmt = lambda v: "(" + ", ".join(f"{int(round(x))}" for x in v) + ")"
print("    diag(A^3) G1/G2 :", fmt(d1), "vs", fmt(d2), "-> differ as multisets")

# (4) signed: balanced vs frustrated triangle
Bal = np.array([[0,1,1],[1,0,1],[1,1,0]], float)
Fru = np.array([[0,1,1],[1,0,-1],[1,-1,0]], float)
print("\n(4) signed Tr(A^3): balanced =", np.trace(Bal@Bal@Bal),
      "| frustrated =", np.trace(Fru@Fru@Fru))
