# N=3 critical-cone local-cap extension

Status: **verified exact-rational theorem; not a global-optimality proof.**

## Objective

The existing directional certificate proves strict local optimality in the
full ordered six-dimensional source space for

\[
0<\lVert A-A^*\rVert_\infty\le 10^{-4}.
\]

The present calculation targets radius `1e-3`.  This is not merely a cosmetic
improvement: source boxes in the global `P3 <= 7.57` branch-and-bound tree reach
width `O(1e-3)` much earlier than width `O(1e-4)`, so a ten-times larger local
cap may change whether the global tree terminates at a useful size.

## Critical and transverse directions

At the symmetric KKT root there are five active observer branches.  Let
`g1,...,g4` be the corner branches, `g5` the moving bottom-edge branch, and

\[
D=\begin{bmatrix}
\nabla g_1-\nabla g_5\\
\vdots\\
\nabla g_4-\nabla g_5
\end{bmatrix}.
\]

The two-dimensional critical space is `ker(D)`.  The new terminal treats the
larger cone

\[
\lVert Du\rVert_2\le 0.3\lVert u\rVert_2.
\]

For the positive KKT-weighted envelope Hessian `H(A)`, the exact computation
uses the S-lemma matrix

\[
H(A^*)-\mu D^TD+\mu(0.3)^2I,
\qquad \mu=2.9336.
\]

If this matrix remains negative definite after adding a uniform Hessian
variation bound, then every nonzero direction in the cone has strictly
negative weighted curvature.

## Exact Hessian-variation bound

`square_riesz/interval_jet3.py` implements third-order automatic
differentiation with `fractions.Fraction` interval endpoints.  The four fixed
corner tensors are differentiated directly.  For the moving observer
`x(A)`, the code differentiates the envelope Hessian

\[
g_{ij}=U_{ij}-\frac{U_{ix}U_{xj}}{U_{xx}},
\qquad x_k=-\frac{U_{xk}}{U_{xx}}.
\]

The full six-dimensional radius-`1e-3` source cube is divided into `2^6=64`
exact rational subboxes.  On every subbox the stationary bottom observer is
bracketed by exact signs of `U_x`; its maximum bracket width is
`0.005934190300703540`.  Taking componentwise third-derivative maxima and a
mean-value bound gives

\[
\lVert H(A)-H(A^*)\rVert_2
\le 1.056728680784948458.
\]

The last step uses symmetry of the Hessian: its spectral norm is at most the
maximum absolute row sum.  Therefore it is sufficient to add the scalar
bound times the identity, rather than independently perturb every matrix
entry.

All six leading principal minors of the negative adjusted matrix have strict
positive lower bounds:

```text
1.264401187470003564019e3
2.729172836329448488673153e6
9.68824817856872592835684119e8
9.91121944604915347058784066733e11
8.18032433784437948932566971142e11
1.6274372415669812515598369643709e13
```

Thus the critical-cone curvature statement is closed by exact rational
interval arithmetic.  The generator is
`scripts/analyze_n3_hessian_lipschitz_exact.py`; an independent recomputation
entry point is `scripts/verify_n3_critical_cone.py`.

## Complementary direction cover

Outside the critical cone, an RTX 5090 binary64 interval-formula search proposed
a cover of all 12 signed `L_infinity` direction faces with 260,234 leaves and
maximum depth 55.  The split coordinate is chosen deterministically by
`direction width x gradient-difference sensitivity`.  The proposal contains:

- 234,420 single-branch leaves;
- 2,238 ordinary weighted-curvature leaves;
- 23,576 critical-cone leaves.

The GPU decisions are untrusted.  Exact verification reconstructed every
rational direction box, checked prefix-free paths and exact Kraft sum one for
each face, and recomputed each terminal inequality.  Full replay completed in
`1025.91` seconds with zero failures.  Exact replay found 223,749 branch leaves,
2,210 ordinary weighted-curvature leaves, and 34,275 critical-cone leaves; an
untrusted candidate label may change when a stronger exact terminal is found.
The smallest exact margin rendered as binary64 for diagnostics is
`1.47093018631455e-7`; floating point is not used for any sign decision.

Consequently, for the unique KKT root `A*`,

\[
0<\lVert A-A^*\rVert_\infty\le 10^{-3}
\quad\Longrightarrow\quad I(A)<I(A^*).
\]

The center-based radius stored in the certificate is exactly `1e-3-1e-40`.
This enlarges the previous explicit full-six-dimensional radius by a factor of
10.

## Global-proof ROI diagnostic

Using the new cap in the float64 global source-box prototype changes the
`P3 <= 7.57` tree from growth to decay.  The number of active boxes at complete
six-level cycle endpoints was

```text
level 53: 435,167
level 59: 257,755
level 65:  17,142
level 66:   5,597
```

The run stopped at its declared depth rather than claiming a proof, but the
cycle multiplier is now below one.  Thus the enlarged cap is algorithmically
material, not just a prettier local constant.

A proposed cap-boundary union terminal was also tested.  After conditioning on
boxes already closed by ordinary witnesses, it added only 8 closures among
15,625 nearby level-48 boxes at target `7.57` (16 near the KKT level).  It is
therefore not worth extending into the exact global verifier in its current
form.

## Scope

The result is a quantitative strict local theorem, not global optimality of
the `N=3` configuration.  Its immediate
research value is as a rigorous terminal condition for the global source-space
branch-and-bound proof.
