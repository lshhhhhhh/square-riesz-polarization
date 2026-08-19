# N=3 strict local-optimality report

## Theorem claimed

Let `A_*` be the unique reflection-symmetric KKT root enclosed by
`data/certificates/n03_symmetric_kkt_krawczyk.json`. In the full
six-dimensional space of three **ordered** source positions, `A_*` is a strict
local maximizer of

\[
I(A)=\min_{p\in[0,1]^2}\sum_{i=1}^3\lVert p-a_i\rVert^{-2}.
\]

This is a local theorem. It does not prove that `A_*` is the global maximizer
or that `P_3=I(A_*)`. The current separate global result remains

```text
7.56838963 <= P_3 <= 7.58.
```

## Computer-assisted prerequisites

The theorem composes three exact-rational certificates:

1. `n03_symmetric_kkt_krawczyk.json` encloses a unique root of the seven
   symmetric equal-value, stationarity, and weight-normalization equations;
   all three orbit weights are positive.
2. `n03_active_minima_isolation.json` proves that the root's global observer
   minimizers are exactly the four corners and `(1/2,0)`, with a strict spatial
   gap away from their five rectangles and strict derivative margins inside.
3. `n03_strict_local_optimality.json` proves LICQ and the nonlinear-programming
   second-order sufficient condition after eliminating the moving bottom-edge
   observer.

All proof decisions use `fractions.Fraction`. Decimal numbers in the JSON are
diagnostics rendered from exact rational endpoints.

## Local five-branch reduction

Write `g_1,...,g_4` for the potentials at the four corners. Strict bottom-edge
curvature and the implicit-function theorem give a unique smooth stationary
observer `x_m(A)` near `1/2`; write

\[
g_5(A)=U((x_m(A),0);A).
\]

The strict derivative and complement margins in prerequisite 2 persist under
sufficiently small arbitrary six-dimensional source perturbations. Therefore,
in some neighborhood of `A_*`,

\[
I(A)=\min_{1\le j\le5}g_j(A).
\]

No explicit neighborhood radius is claimed; strict finite-cover inequalities
and continuity establish its existence.

## Epigraph KKT and LICQ

Consider the smooth local epigraph problem

\[
\max_{A,t}\ t\quad\text{subject to}\quad t-g_j(A)\le0,
\qquad j=1,\ldots,5.
\]

Split the symmetric orbit multipliers equally between reflected corners. The
five individual weights are positive and sum to one. The three symmetric
stationarity equations imply full six-dimensional stationarity: reflection
makes the two paired source gradients mirror each other, while the axis
source's horizontal component vanishes.

The four rows

\[
\nabla g_j-\nabla g_5,\qquad j=1,\ldots,4,
\]

form a `4 x 6` matrix. The exact interval verifier selects the columns

```text
source_1_x, source_1_y, source_2_y, source_3_x
```

and proves their determinant lies near

```text
[-301377.4198543975942053712303134403404762,
 -301377.4198543975942053712303134403404724].
```

Thus the difference matrix has rank four. Adding the common epigraph
`t`-column makes the five active constraint gradients linearly independent,
which is LICQ. The critical source-direction space has dimension two.

## Moving-observer Hessian

The fifth branch cannot be treated as a fixed midpoint witness. Its value
Hessian is the envelope/Schur-complement expression

\[
\nabla^2g_5
=U_{AA}-U_{Ax}U_{xx}^{-1}U_{xA}.
\]

The exact observer-curvature interval is approximately

```text
[35.62283759707285189547886640346840371505,
 35.62283759707285189547886640346840371525],
```

so the implicit elimination is valid.

Using an exact interval graph basis `Z` for the two-dimensional critical
space, the certificate encloses

\[
B=Z^T\left(\sum_j\lambda_j\nabla^2g_j\right)Z
\]

by

```text
B00 in [-3.955957410331491825727899095811512324662,
        -3.955957410331491825727899095811512321105]
B01 in [-0.2529798975043667613656082526756831735700,
        -0.2529798975043667613656082526756831730749]
B11 in [-21.25804174557467085133600265838399909451,
        -21.25804174557467085133600265838399909443]
det(B) in [84.03190894400100102035873465564581999036,
           84.03190894400100102035873465564582006652].
```

Hence `B` is strictly negative definite. Equivalently, the Lagrangian Hessian
for minimizing `-t` is positive definite on the critical cone. With LICQ and
strict complementarity, the standard second-order sufficient condition gives
a strict local optimum of the epigraph problem and therefore of `I(A)`.

## Reproduction

In a clean checkout:

```bash
python scripts/certify_n3_symmetric_kkt.py \
  --candidate data/candidates/n03_symmetric.json \
  --radius 1e-40 \
  --output /tmp/n03-krawczyk.json

python scripts/certify_n3_active_minima.py \
  --kkt-certificate /tmp/n03-krawczyk.json \
  --gap 1/10 \
  --output /tmp/n03-active-minima.json

python scripts/certify_n3_local_optimality.py \
  --kkt-certificate /tmp/n03-krawczyk.json \
  --active-minima-certificate /tmp/n03-active-minima.json \
  --output /tmp/n03-local-optimality.json
```

Expected final diagnostics include:

```text
status: VERIFIED
observer_xx lower: 35.6228375970...
projected b00 upper: -3.9559574103...
projected b11 upper: -21.2580417455...
projected determinant lower: 84.0319089440...
```

The full test suite also independently checks that the graph-basis generalized
eigenvalues reproduce the earlier SVD diagnostic `(-20.90257,-1.62793)`.

## Suggested adversarial review

The highest-value checks are:

1. verify the sign and block structure of the source-coordinate Hessian of
   `1/||p-a_i||^2`;
2. verify the sign of the Schur-complement correction for a minimized observer
   coordinate;
3. check the interval adjugate/inverse indexing and the graph-basis equation;
4. independently derive why rank four of the gradient differences implies
   rank five after the epigraph column is added;
5. check that positive multipliers reduce the critical cone to those five
   active linearized equalities;
6. verify the continuity/implicit-function step that promotes the root-space
   active-set certificate to a full source-neighborhood representation;
7. reject any wording that upgrades strict local optimality to global
   optimality.

The global equality problem remains the higher-risk step. The local theorem is
useful as a rigorous cap for a future global source-space branch-and-bound near
the candidate, but it does not by itself improve the numerical upper bound
`7.58`.

## Quantitative follow-up

The later directional-cover certificate upgrades the existential neighborhood
in this report to the explicit full six-dimensional bound

\[
0<\lVert A-A^*\rVert_\infty\le10^{-4}
\Longrightarrow I(A)<I(A^*).
\]

See `N3_DIRECTIONAL_LOCAL_CAP_REPORT.md` and
`data/certificates/n03_directional_local_cap_0_0001.json`. This remains a
local result, not a proof of global optimality.
