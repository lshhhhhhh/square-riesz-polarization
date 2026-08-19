# Independent audit report

Date: 2026-08-19

Repository: <https://github.com/lshhhhhhh/square-riesz-polarization>

Results baseline: `eb8f53c9ec6d1bbf40984156621133486f7bcb6f`

Upstream comparison submodule: `2146eba2e3cf8ac7dd873365856b58c5cee0d362`

This document is written for an independent reviewer, including an AI reviewer,
who should assume that the numerical search and the certificate generators may
contain mistakes. It separates theorem-level claims, reproducible computations,
diagnostics, and conjectures. Finding an error is more useful than confirming the
authors' expectations.

The first independent review is preserved verbatim in
`INDEPENDENT_AUDIT_FINDINGS.md`; commit `2ef5eb9` is the immutable pre-remediation
snapshot of its scripts and outputs. This report incorporates the accepted
corrections while retaining the review itself as primary audit evidence.

## 1. Problem and notation

For a literal source configuration

\[
X=(a_1,\ldots,a_N)\in([0,1]^2)^N,
\]

define

\[
I(X)=\min_{x\in[0,1]^2}\sum_{i=1}^N\frac{1}{\|x-a_i\|^2},
\qquad
P_N=\max_X I(X).
\]

The singular value at a source is positive infinity. Sources are interchangeable.
The fixed-configuration lower certificates prove statements about one list of
literal finite-decimal coordinates. Only the global `N=3` upper certificate
quantifies over all source configurations.

## 2. Claims submitted for audit

### 2.1 Fixed `N=3` configuration

The audited decimal coordinates are in
`data/candidates/n03_symmetric.json`:

```text
(0.1198267882563919621, 0.4023192320163492298)
(0.8801732117436080379, 0.4023192320163492298)
(0.5,                   0.9801974656079521770)
```

The claimed fixed-configuration interval is

\[
7.56838963\le I(X)
\le 7.568389640029690214292220397565988\ldots.
\]

The lower endpoint was accepted by two exact-rational box bounds:

| Method | Splits | Maximum depth | Record |
|---|---:|---:|---|
| spectral curvature bound | 141 | 32 | `data/certificates/n03_target_7_56838963.json` |
| componentwise Hessian intervals | 123 | 31 | `data/certificates/n03_componentwise_target_7_56838963.json` |

The upper endpoint is exact evaluation at the rational corner `(0,0)`. For the
published 19-digit literal coordinates, the bottom midpoint is higher by about
`3.19e-19`; the exact five-way degeneracy belongs to the high-precision KKT root,
not to its truncated publication coordinates. The two lower methods have
different Taylor remainder bounds, but they share coordinate parsing, rational
arithmetic, branching logic, and some direct-bound code. They are **not**
independent implementations in the strong clean-room sense.

### 2.2 Global `N=3` bracket

The standalone finite-witness tree claims

\[
P_3\le 7.58.
\]

Together with Section 2.1, the theorem-level bracket claimed by this repository is

\[
\boxed{7.56838963\le P_3\le 7.58}.
\]

The upper tree has 816 unordered initial root boxes and 1,304,240 leaves. The
maximum exact leaf bound reported by the independent standard-library verifier is

```text
7370316709888 / 972337298465
= 7.579999987168342...
```

The compressed certificate is
`data/certificates/n03_global_upper_7_58.zip`. Its hashes are:

```text
ZIP:  31CF762FD9CA678E7FF461AC1643AA12E7A251658A73CECF16AEF98E2C38DCC4
JSON: 6F6F936767A73A548FCAFDD18A4F739AED4AE201B093271009BEC4DEE2E0ACEF
```

The uncompressed JSON is deliberately not tracked because it is 95.9 MiB.

### 2.3 Symmetric `N=3` KKT root

`data/certificates/n03_symmetric_kkt_krawczyk.json` claims that the seven-equation
symmetric epigraph KKT system has exactly one root inside the declared rational
box of radius `1e-40`, and that all three active-orbit weights are positive. The
calculation uses rational interval automatic differentiation, exact Gaussian
elimination, and a Krawczyk inclusion.

This proves existence and uniqueness of a root of that **restricted symmetric KKT
system in that box**. It does not prove that all active minima have been found,
strict local optimality in the full six-dimensional source space, or global
optimality.

### 2.4 Other fixed-configuration records

The repository also claims exact lower bounds for literal decimal configurations:

| `N` | Comparison reference | New proved lower bound | Exact point-witness upper bound |
|---:|---:|---:|---:|
| 3 | Friedman display `7.507+` | 7.56838963 | 7.56838964002969021429... |
| 5 | Friedman display `21.342+` | 22.06 | 22.06308301603677... |
| 29 | 272.495973646473 | 282.8 | 282.85692528612702... |
| 30 | 285.326742383550 | 285.34 | 285.34568532988481... |
| 31 | 294.208893270426 | 305.2 | 305.29836691152415... |
| 32 | 304.280799129879 | 317.1 | 317.20381892782922... |
| 33 | 311.665641329667 | 330.5 | 330.59547948010763... |
| 34 | 323.409928097309 | 337.8 | 337.90623582303798... |
| 35 | 329.708966808636 | 347.1 | 347.19572233129303... |

For `N=29..35`, the comparison values are exact point-witness upper bounds for
the configurations in the pinned public upstream repository, conservatively
rounded upward at `1e-12`, not merely rounded values on the Friedman web page.
The exact fractions are pinned in the result CSV and candidate records. Therefore
a proved new lower bound above such a witness proves that the new literal
configuration beats that previous literal configuration. The `N=3` and `N=5`
rows only exceed Friedman display values; without the old literal coordinates and
an exact upper witness, they are not stated as strict defeats of the unrounded old
configurations. None of these statements proves a global optimum.

## 3. Explicit non-claims

This repository does **not** currently prove any of the following:

1. The exact value of `P_3`.
2. Global optimality of the displayed `N=3` configuration.
3. Full strict local optimality of that configuration.
4. Global or local optimality for `N=5` or `N=29..35`.
5. That a numerical continuous minimizer list is exhaustive.
6. That the two fixed-configuration lower methods are clean-room independent.
7. That the records have been accepted by Erich Friedman's page or peer review.

The numerical envelope-Hessian eigenvalues and active-minimum lists in the
research notes are diagnostics supporting future work, not certified theorems.

## 4. Clean-clone setup

A known-good development environment was:

```text
Python 3.14.3
NumPy 2.5.2
SciPy 1.18.0
PyTorch 2.9.0+cu128
CUDA 12.8
NVIDIA GeForce RTX 5090
```

The exact fixed-configuration and global-upper replays do not require a GPU.
PyTorch is needed for GPU search tests and reproduction of the search stage.

```bash
git clone --recurse-submodules --branch agent/independent-audit-report \
  https://github.com/lshhhhhhh/square-riesz-polarization.git
cd square-riesz-polarization
python -m venv .venv
# Activate the environment using the platform-specific command.
python -m pip install -r requirements-audit.txt
python -m unittest discover -s tests -v
```

Expected CPU-only result at publication: 22 tests pass and 4 PyTorch search tests
are explicitly skipped. To reproduce the GPU search environment, install
`requirements-search.txt`; with PyTorch available, all 26 tests pass. A reviewer
should record the platform, Python implementation, dependency versions, wall
times, skips, and any warning.

## 5. Highest-priority standalone audit: global `P_3 <= 7.58`

Extract and hash the certificate:

```bash
cd data/certificates
sha256sum n03_global_upper_7_58.zip
unzip n03_global_upper_7_58.zip
sha256sum n03_global_upper_7_58.json
cd ../..
```

Then run the standard-library verifier; NumPy, SciPy, PyTorch, and the generator
are not imported by this verifier:

```bash
python scripts/verify_global_upper_certificate.py \
  data/certificates/n03_global_upper_7_58.json
```

Expected summary:

```text
status: VERIFIED
target: 379/50
root_count: 816
leaf_count: 1304240
maximum_leaf_bound_decimal: about 7.579999987168342
```

The verifier should be read before it is run. In particular, independently check:

1. `C(16+3-1,3)=816` unordered triples of the 16 initial cells cover every
   three-source configuration modulo source permutation, including coincident
   source cells.
2. Each path reconstructs dyadic child boxes with coordinate schedule
   `depth mod 6`.
3. Leaf paths are unique, prefix-free, and recursively cover their complete root.
4. A witness outside every source box gives the uniform exact bound

   \[
   U(w,B)=\sum_i 1/\operatorname{dist}(w,B_i)^2.
   \]

   For every source choice in `B`, the continuous minimum is at most its value at
   `w`, which is at most `U(w,B)`.
5. The verifier rejects a witness intersecting a source box, an omitted root,
   deleted or duplicated leaves, malformed paths, noncanonical JSON types, and a
   target smaller than a leaf bound.
6. Metadata such as generator status, reports, and split schedule is not trusted.

A particularly valuable audit would reimplement this verifier from the JSON
schema without copying its code. The first audit did so; its original verifier is
preserved at commit `2ef5eb9`. The current
`scripts/verify_global_upper_certificate_cleanroom.py` hardens that implementation
by removing optimization-sensitive `assert` checks and comparing leaf bounds by
exact integer cross-multiplication rather than binary floating point.

## 6. Fixed-configuration lower replay

The fixed-coordinate run records do not contain all leaf boxes. They should be
treated as deterministic exact-computation records, not standalone proof objects.
The proof must be replayed by rerunning the certifier or by independently
reimplementing its inequalities.

Example `N=3` spectral replay:

```bash
mkdir -p audit-output
python scripts/certify_candidate.py \
  --input data/candidates/n03_symmetric.json \
  --target 7.56838963 \
  --witness-x 0 --witness-y 0 \
  --method spectral --max-splits 2000000 \
  --output audit-output/n03-spectral.json
```

Componentwise replay:

```bash
python scripts/certify_candidate.py \
  --input data/candidates/n03_symmetric.json \
  --target 7.56838963 \
  --witness-x 0 --witness-y 0 \
  --method componentwise --max-splits 2000000 \
  --output audit-output/n03-componentwise.json
```

The required result is `"certified": true`. Compare exact rational fields, not
only displayed decimal diagnostics.

The central inequalities are in `square_riesz/certify.py`. Audit them directly:

1. Literal decimal strings must become exact `fractions.Fraction` values.
2. On a query box, `1 / maximum_distance_squared` is a valid per-source lower
   bound, including boxes that contain a source singularity.
3. The spectral Taylor bound uses the per-source eigenvalue inequality
   `lambda_min(H) >= -2/r^4` away from the source.
4. The componentwise method independently intervals `H_xx`, `H_yy`, and `H_xy`
   and applies a lower quadratic remainder.
5. Every failed box is bisected along the longer spatial axis; termination
   with an empty heap therefore covers the original full unit square.
6. The exact point witness supplies only an upper bound for `I(X)` and must never
   be used to justify the full-square lower bound.

The tests replay the pinned upstream `N=7` result with the generalized code. This
is a regression test, not a substitute for checking the inequalities.

For `N=29..35`, candidate paths, targets, witness points, and recorded outputs are
listed in `data/candidates/` and `data/certificates/`. Replaying one high-`N` case
from each method is recommended before attempting all fourteen runs.

## 7. Krawczyk replay

```bash
python scripts/certify_n3_symmetric_kkt.py \
  --candidate data/candidates/n03_symmetric.json \
  --radius 1e-40 \
  --output audit-output/n03-krawczyk.json
```

Expected conditions are `strictly_inside=true` and `weights_positive=true`.
Independently verify the seven equations in `kkt_system`, the ordering of the
variables, the rational Jacobian inverse, interval matrix multiplication, and the
strict interior test. Also verify that the KKT system corresponds only to the
declared symmetric active-orbit model.

## 8. Numerical search provenance

The RTX 5090 searcher is `scripts/record_hunt.py`. It overlaps CUDA candidate
generation with CPU continuous minimization, atomically persists jobs, and keeps
the historical best hard-minimum member so later soft-minimum steps cannot erase
a strong incumbent.

The large-`N` runs used `initialization=incumbent-jitter`, `jitter=0.001`, and the
literal coordinates from the pinned upstream submodule. They are local refinement
runs, not independent rediscoveries from random starts; accepted candidates can
nevertheless move individual sources by roughly `0.09`. The upstream field
`minimum_source_separation` is documented as measured metadata, not an optimizer
constraint or a proof assumption. Near-coincident sources in some older
incumbents remain a useful structural diagnostic, but do not justify a claim that
this project removed a hidden separation constraint.

The formal large-`N` batch is in:

```text
runs/record_hunt_large_20260819/
```

It contains 24 GPU jobs for `N=29,31,32,33,34,35`, four deterministic seeds per
`N`, and CPU evaluation of the top 12 candidates per job. The matched `N=30`
anti-selection-bias run is in:

```text
runs/record_hunt_n30_20260819/
```

Search logs establish provenance and help reproduce candidate discovery. They are
not used by the exact fixed-configuration proof after coordinates are frozen.

Audit risks specific to the search stage include failure to find a darker local
minimum, GPU float32 ranking error, dependence on incumbent jitter, seed selection,
and post-selection bias. None of these can create a false certified lower bound if
the fixed-configuration certifier is correct, but they affect scientific claims
about optimization difficulty and typical structure.

## 9. Provenance and comparison baseline

The submodule `upstream/certificates` pins the public repository
`Driedsandwich/square-riesz-polarization-certificates`. Its coordinates and exact
point witnesses define the comparison baseline for covered `N`. The generalized
fixed-configuration certifiers were adapted from its BSD-3-Clause code; see
`THIRD_PARTY_NOTICES.md`.

Reviewers should verify the pinned commit and compare the old configuration's
exact witness directly. Be careful not to compare a new lower bound merely with a
rounded Friedman display value or an old configuration's lower certificate.

## 10. Suggested adversarial tests

Highest value tests, in approximate order:

1. Write a clean-room global-upper verifier and replay the 1,304,240-leaf tree.
2. Delete one leaf, duplicate one path, alter one root, corrupt a witness index,
   and lower the target; every mutation must fail.
3. Reimplement both box lower bounds with a different interval package and test
   random rational boxes against dense/high-precision sampling.
4. Symbolically or automatically differentiate `1/(x^2+y^2)` and compare every
   Hessian interval formula and sign.
5. Replay the `N=3` lower certificate on a non-CPython implementation or with a
   separate exact-rational library.
6. Recompute the exact corner `(0,0)` and bottom-midpoint `(1/2,0)` values directly
   from the three published decimal sources and confirm their tiny ordering.
7. Verify that source-containing boxes never use a Taylor bound across a
   singularity.
8. Check permutation reduction for roots containing repeated initial cells.
9. Attempt to construct a source configuration with score above `7.58`; any such
   configuration would falsify the global certificate or verifier.
10. Search for an unlisted dark point below `7.56838963`; any such point would
    falsify the fixed-configuration certificate or coordinate interpretation.

## 11. Audit outcome template

Please report conclusions separately for:

```text
A. Literal coordinate parsing and model agreement
B. Fixed N=3 lower bound
C. Fixed N=3 exact point upper witness
D. Global P3 <= 7.58 tree coverage and leaf inequalities
E. Symmetric KKT root inclusion
F. N=5 and N=29..35 fixed-configuration lower bounds
G. Search provenance and reproducibility
H. Claim wording / overclaim audit
```

For each item, use one of `verified`, `plausible but not independently verified`,
`failed`, or `out of scope`, and attach the exact command, artifact hash, and first
counterexample if applicable.
