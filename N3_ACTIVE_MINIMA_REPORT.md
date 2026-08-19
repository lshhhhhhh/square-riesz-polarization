# N=3 active-minimum isolation report

## Result under review

Let `(a,b,c,lambda_T,lambda_C,lambda_M,t)` be the unique root enclosed by
`data/certificates/n03_symmetric_kkt_krawczyk.json`, and put the three sources
at

```text
(a,b), (1-a,b), (1/2,c).
```

The exact-rational certificate
`data/certificates/n03_active_minima_isolation.json`, combined with that
Krawczyk prerequisite, proves that the global minimizers of the observer
potential on `[0,1]^2` are exactly

```text
(0,0), (1,0), (0,1), (1,1), (1/2,0).
```

It does **not** prove that the source configuration is a local or global
maximizer of `P_3`.

## Reproduction

Only Python's standard library is needed by this replay:

```bash
python scripts/certify_n3_symmetric_kkt.py \
  --candidate data/candidates/n03_symmetric.json \
  --radius 1e-40 \
  --output /tmp/n03-krawczyk.json

python scripts/certify_n3_active_minima.py \
  --kkt-certificate /tmp/n03-krawczyk.json \
  --gap 1/10 \
  --output /tmp/n03-active-minima.json
```

Expected structural output:

```text
status: VERIFIED
complement splits: 35
corner derivative splits: 0, 0, 2, 2
bottom-midpoint derivative splits: 1
```

Elapsed-time fields and the prerequisite file hash can change when the
Krawczyk artifact is regenerated; the exact inequalities and split counts do
not.

## Proof decomposition

### 1. Prerequisite root box

The prior Krawczyk certificate proves existence and uniqueness of the seven
variable symmetric KKT root in a rational box of radius `1e-40`. In particular,
at that root the top-corner, bottom-corner, and bottom-midpoint potentials all
equal the root variable `t`.

The active-minimum verifier reads the exact rational center and radius from
that artifact. For potential and derivative inequalities it deliberately
forgets correlations between the source-coordinate intervals; this enlarges
the parameter set and is conservative.

### 2. Four corner rectangles

On each `1/8 x 1/8` corner rectangle, interval derivatives prove strict
coordinate monotonicity toward the corresponding corner:

| rectangle | signs |
|---|---|
| bottom left | `U_x>0`, `U_y>0` |
| bottom right | `U_x<0`, `U_y>0` |
| top left | `U_x>0`, `U_y<0` |
| top right | `U_x<0`, `U_y<0` |

Thus the unique minimum in each rectangle is its corner. The weakest exact
certified coordinate-derivative margin is about `0.484139`.

### 3. Bottom-midpoint rectangle

On `[3/8,5/8] x [0,1/8]`, interval derivatives prove `U_y>0` and `U_xx>0`.
The exact reflection symmetry gives `U_x(1/2,0)=0`. Therefore the potential
first decreases and then increases along the bottom edge, while increasing
strictly into the square; `(1/2,0)` is the rectangle's unique minimum.

The exact certified margins are about `10.890897` for `U_y` and `4.907573` for
`U_xx`.

### 4. Spatial complement

The remaining part of the unit square is covered by rational dyadic boxes.
Every accepted leaf satisfies

```text
U >= level_center + root_radius + 1/10.
```

Since the true root level is at most `level_center + root_radius`, the
complement lies strictly more than `1/10` above the five equal KKT values. The
cover closes after 35 splits; its weakest computed leaf lower bound is about
`7.69088827396077`, above the target near `7.66838964002969`.

For a varying source box, the lower bound is the maximum of:

1. a direct reciprocal of the maximum possible squared distance for each
   source; and
2. the existing exact fixed-center spectral bound minus an exact source-motion
   Lipschitz allowance.

If an observer box intersects a source box, only the always-valid direct bound
is used.

## Exact-arithmetic trust boundary

- `square_riesz/local_proof.py` contains the rational interval operations,
  derivative enclosures, source-motion allowance, and exhaustive covers.
- `scripts/certify_n3_active_minima.py` fixes the five rational rectangles,
  composes the results, and writes the review artifact.
- All decisions that mark a leaf as proved compare `fractions.Fraction`
  objects. Floating-point values appear only as human-readable decimal
  diagnostics and elapsed time.
- CI first regenerates the Krawczyk inclusion and then feeds that fresh
  prerequisite into the active-minimum replay.

## Suggested adversarial review

An independent reviewer should focus on:

1. the Lipschitz allowance in `varying_source_box_lower_bound`;
2. completeness of the recursive unit-square cover after excluded rectangles
   are removed;
3. sign orientation at the four constrained corners;
4. the inference from `U_y>0`, `U_xx>0`, and reflection symmetry in the
   bottom-midpoint rectangle;
5. the dependency on the KKT equal-value equations;
6. whether any claim accidentally upgrades observer-space isolation into
   source-space local optimality.

The next mathematical stage is separate: certify the full six-dimensional
epigraph KKT regularity and the negative-definite moving-minimum envelope
Hessian. Until then the local-optimality theorem remains open.
