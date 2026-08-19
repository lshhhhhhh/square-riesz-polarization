# N=3 quantitative directional local cap

Status: **verified exact-rational theorem; not a global-optimality proof.**

## Result

Let `A*` be the unique reflection-symmetric KKT root enclosed by
`data/certificates/n03_symmetric_kkt_krawczyk.json`. For ordered triples of
sources in the full six-dimensional source space, the certificate proves

\[
0<\lVert A-A^*\rVert_\infty\le 10^{-4}
\quad\Longrightarrow\quad I(A)<I(A^*).
\]

Since the exact root lies within `1e-40` of the stored rational center, the
corresponding center-based certified radius is exactly `1e-4-1e-40`. This
improves the earlier two-cone Taylor radius `7.3298064197e-6` by a factor of
about `13.64`.

This theorem is strictly local. It does not change the current global bound
`P3 <= 7.58` and does not prove that no better configuration exists outside
the certified box.

## Why a direction cover works

Write every nonzero displacement as `A-A*=t u`, where
`t=||A-A*||_infinity` and `u` lies on one of the 12 signed faces of the
six-dimensional unit box. Each face has five free coordinates. The
certificate recursively bisects these free coordinates in a deterministic
cyclic order.

At the root there are five equal observer branches: the four fixed corners
and the moving local minimum on the bottom edge near `(1/2,0)`. For every
direction box, exact interval arithmetic encloses all source configurations
on all represented rays for `0 <= t <= 1e-4`, together with the gradients and
Hessians of these five branches.

### Terminal type 1: one branch decreases

For a branch `g_j`, Taylor's theorem gives

\[
\frac{g_j(A^*+tu)-g_j(A^*)}{t}
\le \nabla g_j(A^*)\cdot u
+\frac{t}{2}\,u^T H_j(\xi)u.
\]

If the interval upper bound

\[
\sup \nabla g_j(A^*)\cdot u
+\frac{10^{-4}}2\max\!\left(0,\sup u^TH_j u\right)<0,
\]

then that observer branch is below the root value along every nonzero ray in
the direction box. Since the continuous square minimum is no greater than the
value at any selected observer, this closes the box.

### Terminal type 2: the KKT weighted average curves downward

Let the five positive KKT multipliers be `lambda_j`. They sum to one, the
weighted gradient vanishes, and all five branches agree at the root. Hence

\[
\min_j(g_j(A)-g_j(A^*))
\le \sum_j\lambda_j(g_j(A)-g_j(A^*)).
\]

If exact interval arithmetic proves

\[
\sup u^T\left(\sum_j\lambda_j H_j(\xi)\right)u<0,
\]

then the weighted average, and therefore at least one branch, decreases along
the entire represented ray. This closes boxes near the two-dimensional
critical direction space where no individual first derivative has a uniform
negative sign.

## Moving bottom observer

The bottom observer is not frozen at `x=1/2`. The full-source neighborhood
certificate proves `U_xx>3` throughout the relevant bottom rectangle and
brackets a unique stationary bottom observer. For each direction box the
verifier encloses `U_x(1/2,0)` and uses

\[
|x_m-1/2|\le |U_x(1/2,0)|/3.
\]

It then evaluates the envelope/Schur-complement Hessian

\[
H_m=U_{AA}-U_{Ax}U_{xx}^{-1}U_{xA}
\]

over that observer interval. Freezing the observer would be invalid and gives
a wrong positive critical direction.

## Certificate statistics

- 12 signed direction-face roots;
- 47,621 exact terminal leaves;
- depth range `0..44`, average depth about `27.10`;
- 41,433 single-branch leaves and 6,188 weighted-curvature leaves;
- exact replay failures: `0`;
- full 16-process replay time: `1878.9 s`;
- smallest exact margin rendered as binary64 for diagnostics:
  `1.0706040811515514e-6`.

The last number is not used as a floating-point proof. Every sign decision is
made with `fractions.Fraction` endpoints.

## Trust boundary

The RTX 5090 generated a candidate binary direction tree using binary64
interval-formula diagnostics. The final certificate does not trust those
decisions and does not require the candidate tree. It stores the final leaf
paths and exact terminal type. The verifier independently:

1. checks the KKT and active-neighborhood canonical JSON hashes;
2. rejects malformed, duplicate, nonbinary, or prefix-colliding paths;
3. checks the exact Kraft sum is one for each of the 12 roots;
4. reconstructs every rational direction box;
5. recomputes every derivative, observer interval, Schur complement, and
   terminal inequality with exact rational arithmetic;
6. checks that the recomputed terminal type and branch match the certificate.

The published artifact is
`data/certificates/n03_directional_local_cap_0_0001.json`. A full replay is:

```bash
python scripts/verify_n3_directional_local_cap.py --workers 16
```

For a quick wiring check, append `--limit 20`; that checks the full tree
coverage metadata and exactly replays the first 20 leaves, but reports only
`SAMPLE_VERIFIED`.

## ROI and next step

The old scalar two-cone inequalities lost too much information by replacing
directional quadratic forms with entrywise norms. GPU searches found the
worst boundary decrease at radius `1e-3` to be about `1.46e-5` and at radius
`3e-3` about `1.07e-4`, so the mathematics suggests a substantially larger
basin. However, the current interval dependency tree for radius `0.003`
exceeded one million active boxes by depth 21. Therefore:

- `1e-4` is a completed, high-ROI theorem;
- pushing the same isotropic tree directly to `0.003` is currently poor ROI;
- the next proof improvement should parameterize the two-dimensional critical
  manifold and transverse directions separately, rather than raise node
  limits.
