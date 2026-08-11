# The Functional Symbiotic Resilience Index

**Topological Entropy, Wasserstein Curvature Bounds, and Non-Equilibrium Thermodynamics of Complex Networks**

**Alberto Acedo** · Biome Makers Inc. · acedo@biomemakers.com  
[![Patent Pending](https://img.shields.io/badge/USPTO-Patent%20Pending%2064%2F121%2C656-blue)](https://www.uspto.gov)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-red)](https://arxiv.org)

---

## What This Is

A mathematical framework connecting the **Functional Symbiotic Resilience Index** (FSRI, Ω) to non-equilibrium thermodynamics and discrete differential geometry.

**Central result:** The topological pressure functional `T(W) = Tr(WWᵀ)³` is simultaneously:
1. Proportional to the global clustering coefficient of the induced weight graph
2. The generator of a gradient flow whose fixed points are Prigoginian minimum-entropy-production states
3. Related to total positive Ollivier-Ricci curvature by a proven upper bound (the matching lower bound is not established in general; see Remark 8 of the paper)

This provides the theoretical foundation for the [Omega-S neural network regularizer](https://arxiv.org/abs/2608.03887).

---

## Key Results

| Result | Statement |
|--------|-----------|
| **Gradient theorem** | ∂Tr(A³)/∂A = 3A² (exact) |
| **ORC upper bound** (proven) | Σκ⁺(x,y) ≤ Tr(A³)/(2d_min), satisfied on all six test networks |
| **Exact WS formula** | Ω(K,p) = [(3K-6)/(4K-2)] · (1-p)² / (Np), r=0.9997 vs numerical |
| **Exact thermodynamic relation** | Ω = [(3K-6)/(4K-2)] · S²_env / (Np · E³_ext) |
| **Phase transition** | Tr(A³) increases 76× over triangle-density phase transition |
| **Gini discriminator** | Monopoly networks: G=0.986, Uniform networks: G=0.272 (3.6× difference) |
| **Spectral identity** | Tr(A³) = Σλ³: the global functional is a spectral statistic, not an alternative to one |
| **Locus of the contribution** | diag(A³) is *not* determined by the spectrum (explicit cospectral counterexample) |
| **Order dependence** | Ω carries an explicit 1/n; the shape-normalised variant removes it (factor 47 → 1.1 per decade of sampling effort) |

---

## What the applied work found, and what it costs this framework

The companion paper on LLM fine-tuning
([arXiv:2608.03887](https://arxiv.org/abs/2608.03887)) measures the
framework rather than assuming it, and two of its findings bear directly on
what is claimed here. We list them because they are the kind of thing a reader
should not have to discover for themselves.

**The clustering factor is inert under the construction used there, and
reviving it makes things worse.** Under the pseudo-adjacency
`A = σ(|WWᵀ|)` the measured triadic excess `C/D` is exactly 1.0000, equal to
its permutation null, in every module tested: the constructed graph carries no
triadic content over its density at all. Section 2.1 of the paper (Remark 5)
now reports this with the mechanism. An earlier version of that remark offered
a contrast-preserving construction as the obvious repair; it has since been
built and evaluated, it does what it was designed to do, and **retention gets
worse on all ten seeds**. The negative result is recorded rather than left as
an open suggestion.

**The implementation branches on matrix shape.** The reference code forms
`σ(|WWᵀ|)` only when `W` is non-square and applies `σ(|W|)` elementwise when it
is square. The degree-variance factor differs by roughly 400× between the two
branches on the square module measured. This is declared in both papers; it
does not invalidate any number, and it does mean a reimplementation should fix
the branch deliberately.

## `M` is now fixed, and where the orientation comes from

Earlier statements of Definition 1 gave `M` as *"a modularity measure (e.g.,
inverse spectral gap)"*. Because `M` sits in the denominator, that latitude
inverts how the index responds to compartmentalisation, so **Definition 1 now
fixes `M = 1/lambda2(L)`**.

The orientation is not a convention. It is fixed by the observation the index
was built to formalise. In [Ortiz-Alvarez et al., *mSystems*
2021](https://doi.org/10.1128/msystems.00344-21), a survey of 350 vineyard
soils, the low-intervention systems, which are the ones associated with
small-world structure and resistance to perturbation, show simultaneously
**higher clustering, lower modularity and lower co-exclusion**. Every factor
inherits its direction from that observation, `M` included: for the index to be
large on those communities it must decrease with modularity, so `M` must be a
modularity measure, and `1/lambda2`, which is large on graphs that separate
easily, is the correct choice.

## Relation to the applied implementations

The [LLM regulariser](https://arxiv.org/abs/2608.03887) computes a
related but distinct objective: it differs in the construction of `A`, in the
normalisation of `C` and in the orientation of `M`. That repository documents
each difference and measures which factors are live under its construction. We
note the relation rather than the detail, because **no result of this paper
depends on it**: Propositions 3, 4, 5 and 7 and the ORC bound are statements
about `Tr(A^3)` and about individual factors, not about how the four are
composed. A financial companion paper places `Coex` on the other side of the
ratio; that difference is now settled, and is explained below.

One point does belong here because it concerns the index rather than any
implementation. Under the bounded map used to build `A` from weight matrices,
three of the four factors go numerically inert: elasticity at or below 1e-4,
against 9e-3 for `Coex`. **That is a property of that construction, not of the
index.** In the ecological setting the index comes from, modularity is among
the strongest discriminators available (r2 = 0.955 against management type,
comparable to co-exclusion and higher than clustering). A factor that carries
most of the signal where it was born and none under a saturating map on neural
weights is telling us about the map.

| composition | `M` | position of `Coex` |
|---|---|---|
| this paper, as now fixed | 1/λ₂ | denominator |
| financial companion paper | 1/λ₂ | **numerator** |
| LLM regulariser library | **λ₂** | denominator |

Two differences remain, of different kinds. The financial work agrees on `M`
and places `Coex` in the numerator, and that one is settled rather than
pending: under financial stress every factor moves the same way, correlations
rising so that `C`, `D` and `Coex` all increase while the inverse spectral gap
falls. An indicator built to *rise* under stress must therefore compose the
same four factors differently from an index built to be *large* on resilient
systems. Beneath that sits an asymmetry worth stating: a densely
interconnected, weakly modular community is the healthy configuration in the
ecological setting this index comes from, and the crisis configuration in a
market. The index as defined is oriented for the former, and applying it
unchanged to the latter would call a crash resilient.

**The regulariser library differs on `M`, and the difference is one of
direction.** Its power iteration estimates λ₂ rather than 1/λ₂ and drives it
*down*, while Definition 1 requires λ₂ *up* for large Ω. Read on its own that is
an unresolved conflict.

**It is resolved by measurement, in the direction that makes it immaterial.**
Elasticity of each factor with respect to the weights, median over ten attention
projections of an 8B model spanning layers 0 to 31:

| `C` | `D` | `M` | `Coex` |
|---|---|---|---|
| 0.0000 | 0.0000 | 0.0001 | **0.0091** |

Three of the four factors are numerically inert; the modularity term moves about
ninety times less than degree variance. **The orientation of `M` has therefore
had no effect in either direction**, because the factor does not respond to the
weights at all under that construction. What needs correcting is the description
of the objective, not its behaviour.

Two things follow. As implemented, the applied objective reduces in practice to
a penalty on `Coex` alone, which is the factor this framework identifies with
structural monopoly and whose direction is not in dispute: the empirical results
are obtained through the channel the framework predicts, even though the
composite is not maximising Ω as defined here and should not be described as
doing so. And a construction under which `M` is live would make the orientation
matter, so testing whether correcting it improves retention is an experiment
nobody has run.

**None of this paper's results depends on the resolution.** Propositions 3, 4,
5 and 7 and the ORC bound are statements about `Tr(A³)` and about individual
factors, not about how the four are composed.

A separate point concerns the *estimator* rather than the quantity. The applied
implementation obtains λ₂ by three steps of mean-centred shifted power
iteration, which is few: on a graph with clear community structure and
λ₂ = 7.83 it returns 19.27, still between λ₂ and λ_max. The estimate is biased
upward and the bias depends on the graph's spectrum. This does not affect the
measured behaviour of the regulariser, where the operative factor is `Coex`,
but any use of `M` as a reported number rather than as a gradient source should
iterate further or report convergence.

---

## Computational Validation (CPU only)

| Notebook | What it validates | Runtime |
|-------|------------------|---------|
| `colabs/fsri_colab1_topologia.py` | Ω on one draw from each synthetic family: diverges on the regular ring, and shows that Tr(A³) alone does *not* order the families by monopoly. Single-seed, so it illustrates Proposition 2 rather than reproducing its 40-seed medians | ~2 min |
| `colabs/fsri_colab2_termodinamica.py` | Phase transition; Prigogine gradient; entropy production | ~3 min |
| `colabs/fsri_colab3_orc_ws.py` | ORC↔Tr(A³) correlation; exact WS formula (r=0.9997) | ~2 min |
| `colabs/fsri_colab4_real_networks.py` | The ORC bounds on real networks with the fast Jost-Liu approximation. Superseded by 4b for any number that matters | ~3 min |
| `colabs/fsri_colab4b_orc_exact.py` | Exact Wasserstein-1 ORC; generates Table 3 of the paper (deterministic) | ~4 min |
| `colabs/fsri_colab5_spectral_identity.py` | Spectral identity and the cospectral counterexample (detail below) | ~1 min |
| `colabs/fsri_colab6_discriminative.py` | Reproduces Proposition 2: the forty-seed medians and ranges of Ω at N=50, K=6, and the divergence on the regular ring | <1 min |
| `colabs/check_M.py` | Elasticity of each of the four factors on a real model's weights; the measurement behind the inert-factor table above | minutes, laptop |

All of them run on CPU with no additional installation required. `check_M.py`
needs an already-downloaded Hugging Face checkpoint and defaults to
Llama-3-8B; pass a different repo id or path to run it on another model.

---

## Installation

```bash
git clone https://github.com/BiomeMakers/OmegaS-fsri.git
cd OmegaS-fsri
pip install networkx numpy matplotlib scipy  # all Colabs
```

---

## Citation

```bibtex
@article{acedo2026fsri,
  title   = {The Functional Symbiotic Resilience Index: Topological
             Entropy, Wasserstein Curvature Bounds, and Non-Equilibrium
             Thermodynamics of Complex Networks},
  author  = {Acedo, Alberto},
  year    = {2026},
  note    = {Patent pending},
  url     = {https://github.com/BiomeMakers/OmegaS-fsri}
}
```

---

## License

**Paper (LaTeX/PDF):** CC BY-NC-ND 4.0 : share and cite freely for non-commercial purposes, with attribution and without redistributing modified versions. Same license as the companion paper on arXiv.  
**Code (Colabs):** AGPL-3.0 for research and non-commercial use, with internal evaluation explicitly permitted; a commercial licence is required for production use. See [`LICENSE`](LICENSE). 
**Patent:** patent pending. Commercial use of the FSRI index in production systems may require a license. Contact: acedo@biomemakers.com

---

## Venue

Target: **Physical Review E** (Interdisciplinary Physics section) or **Journal of Statistical Mechanics**.  
arXiv categories: cs.LG (primary), cross-listed to cs.SI and math-ph

*© 2026 Alberto Acedo · Biome Makers Inc.*


## Section 2.4 : Spectral identity (new)

`colabs/fsri_colab5_spectral_identity.py` verifies numerically the propositions of
Section 2.4:

- **Tr(A³) = Σλᵢ³** : the *global* functional is exactly the third spectral moment,
  placing it in the same family as the Absorption Ratio, effective rank and the Vendi
  score. The "topological vs. spectral" contrast is not available at this level.
- **(A³)ᵢᵢ = Σₖ λₖ³ (vₖ)ᵢ²** : the node-wise decomposition depends on *eigenvectors*.
- **Explicit cospectral counterexample** (6 vertices): identical spectrum and identical
  Tr(A³) = 12, but different diag(A³) : proving the node field is *not* spectrally
  determined. This locates the framework's non-redundant content in the node-wise field
  rather than the global scalar.
- **Signed Tr(A³) = 6·(balanced − unbalanced triangles)** : Heider structural balance,
  information inaccessible to any index computed on |A|.
