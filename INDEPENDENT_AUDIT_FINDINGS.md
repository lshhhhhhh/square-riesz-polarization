# Independent audit findings

Date: 2026-08-19 (UTC)

Auditor: independent AI reviewer, working only from
`INDEPENDENT_AUDIT_REPORT.md` and the repository contents.

Audited tree: branch `agent/independent-audit-report`, commit `dde0fe0`
(results baseline `eb8f53c9ec6d1bbf40984156621133486f7bcb6f`).

Upstream submodule at audit time: `2146eba2e3cf8ac7dd873365856b58c5cee0d362`
— matches the pinned commit declared in the report.

Audit environment (deliberately *not* the development environment):

```text
Windows 11, CPython 3.12
NumPy 2.4.6, SciPy 1.17.1
PyTorch absent, no GPU used
```

Every exact replay below succeeded on this different Python minor version with
no PyTorch present, which is itself evidence that the proof objects do not
depend on the search stack.

## 1. Headline conclusion

**No theorem-level claim in the report was falsified.** The two bracket
endpoints

\[
7.56838963\le P_3\le 7.58
\]

both survive independent re-derivation, exact replay, a clean-room second
implementation, mutation testing, and adversarial numerical attack.

All findings below concern **presentation, provenance, and scientific
interpretation** of the `N=5` and `N=29..35` comparison claims, plus one
tightness issue at the `N=3` upper endpoint. Finding 1 is the only one that
changes how a reader should interpret a result.

## 2. Verdicts against the report's own section 11 template

| Item | Verdict | Basis |
|---|---|---|
| A. Coordinate parsing and model agreement | **verified** | Literal decimals become exact `Fraction`s; unit-square and distinctness guards present; the stored exact rational witness value in all 23 fixed-configuration certificates was recomputed from the published decimal strings and matched in every case |
| B. Fixed `N=3` lower bound | **verified** | Both inequalities re-derived by hand; both replays reproduce the reported split and depth counts exactly; 4,000-case randomized soundness test and 12,000-case Hessian enclosure test both clean |
| C. Fixed `N=3` exact point upper witness | **verified but not tight** | Value confirmed to 40 digits; see finding 3 |
| D. Global `P_3 <= 7.58` tree coverage and leaf inequalities | **verified** | Repository verifier plus an independent clean-room verifier agree on the exact maximum leaf bound; 10 of 10 mutations rejected by both |
| E. Symmetric KKT root inclusion | **verified** | All seven equations and nine gradient components re-derived by hand; Krawczyk operator form correct; byte-identical replay |
| F. `N=5` and `N=29..35` fixed-configuration lower bounds | **verified as worded** | All certificates replay and are internally consistent; but see findings 1, 2, 6 for how the comparison should be read |
| G. Search provenance and reproducibility | **verified structurally** | Job and evaluation counts match section 8; see finding 4 for a disclosure gap |
| H. Claim wording / overclaim audit | **six issues** | Findings 1, 2, 3, 4, 5, 6 |

## 3. What was executed

| Check | Command or method | Result |
|---|---|---|
| Certificate hashes | `sha256sum` on zip and expanded JSON | Both match section 2.2 byte for byte; the zip expands to exactly the declared JSON hash |
| Repository global verifier | `scripts/verify_global_upper_certificate.py` | `VERIFIED`; 816 roots; 1,304,240 leaves; max leaf `7370316709888/972337298465`; 84.5 s |
| **Clean-room global verifier** | `scripts/verify_global_upper_certificate_cleanroom.py` (written from the JSON schema only) | `VERIFIED`; identical exact maximum leaf bound |
| Mutation battery | 10 mutations of the tracked `7.7` certificate | 10 of 10 rejected by **both** verifiers |
| `N=3` lower replay, spectral | `certify_candidate.py --method spectral` | `certified: true`, 141 splits, depth 32 — matches section 2.1 |
| `N=3` lower replay, componentwise | `certify_candidate.py --method componentwise` | `certified: true`, 123 splits, depth 31 — matches section 2.1 |
| High-`N` replay, one per method | `N=33` spectral, `N=34` componentwise | 2,788/17 and 2,751/18 — identical to the stored certificates |
| Krawczyk replay | `certify_n3_symmetric_kkt.py --radius 1e-40` | Byte-identical to the stored certificate; relative inclusion margin 2.6e-38 |
| Box-bound soundness | 4,000 random rational configurations and boxes, each compared against exact sampling on a 12x12 interior grid | 0 violations for either method |
| Hessian enclosure | 12,000 exact pointwise Hessians tested against the componentwise intervals | 0 enclosure failures |
| Singularity handling | Boxes with a source strictly inside, on a corner, and on an edge | Taylor branch correctly suppressed in every case; only the direct bound is used |
| Adversarial dark point | 1,201x1,201 grid plus 120 Nelder-Mead polishes | Darkest 7.5683896400296895 — no point below the certified 7.56838963 |
| Adversarial configuration | 400-member population hill-climb, then a 2,001x2,001 observer grid on the winner | Best 7.5367813 — nothing above 7.58 |

### 3.1 Mathematics re-derived independently

- For the kernel `1/r^2` the radial Hessian eigenvalues are `6/r^4` and
  `-2/r^4`, so `lambda_min = -2/r^4`. The spectral bound's use of
  `curvature * (half_width^2 + half_height^2)` is exactly the half-eigenvalue
  term with the squared displacement bounded by the box half-diagonal.
  Correct, and the sign is conservative.
- `H_xx = (6dx^2 - 2dy^2)/r^6`, `H_yy = (6dy^2 - 2dx^2)/r^6`,
  `H_xy = 8 dx dy / r^6`. The interval numerators and the positive-denominator
  division in `_componentwise_hessian_intervals` are correct enclosures.
- The remainder assembly
  `min(0,Hxx_lo)*hx^2/2 + min(0,Hyy_lo)*hy^2/2 - max|Hxy|*hx*hy`
  is a valid lower bound on the quadratic Taylor remainder for every
  intermediate point in the box.
- The Krawczyk operator `K(X) = m - Y f(m) + (I - Y F'(X))(X - m)` is
  implemented correctly, with `Y` the exact inverse of the point Jacobian, and
  the strict-interior test is the right acceptance condition.
- All seven KKT equations were checked against the geometry of the three active
  observer orbits — top corners, bottom corners, bottom midpoint — and all nine
  partial derivatives with respect to `(a, b, c)` are correct.

### 3.2 The clean-room verifier

`scripts/verify_global_upper_certificate_cleanroom.py` was written from the JSON
schema without reading the repository verifier's control flow, and differs from
it in three ways that matter:

1. **Arithmetic representation.** Scaled integers with explicit denominators,
   never `fractions.Fraction`. A shared bug in rational normalization could not
   affect both.
2. **Coverage proof.** Kraft equality over the leaf path lengths combined with
   trie-based prefix-freeness, instead of the repository's recursive `covered()`
   descent. These are different theorems about the same tree.
3. **Parsing.** The decimal target is parsed by hand into an integer numerator
   and denominator rather than via `Fraction(str)`.

Both implementations return the identical exact maximum leaf bound
`7370316709888/972337298465 = 7.579999987168342...`, and both reject all ten
mutations. This satisfies the report's own section 10 item 1, which was listed
as the highest-value adversarial test.

### 3.3 Mutation battery detail

Every row was rejected by both the repository verifier and the clean-room
verifier:

```text
delete one leaf                    duplicate one leaf path
alter one leaf root                corrupt one witness index
lower the target to 7.0            omit an entire root box
append one path bit                drop one path bit
leaf_count metadata mismatch       target supplied as a JSON float
```

## 4. Findings

### Finding 1 — the large `N=29,31..35` gains are a constraint artifact, and the control run demonstrates it

**Severity: substantive. Affects interpretation, not correctness.**

The upstream `certified-results.csv` records a `minimum_source_separation` of
`0.0050000` — a binding constraint — for exactly the six values
`N=29,31,32,33,34,35`, and `0.1236006` for `N=30`. The new configurations carry
separations of 0.082 to 0.168, and `scripts/record_hunt.py` imposes no
separation floor at all.

| `N` | upstream min separation | new min separation | relative gain |
|---:|---:|---:|---:|
| 29 | 0.0050000 | 0.1676839 | +3.8% |
| 31 | 0.0050000 | 0.1464353 | +3.8% |
| 32 | 0.0050000 | 0.1393581 | +4.2% |
| 33 | 0.0050000 | 0.1397371 | +6.1% |
| 34 | 0.0050000 | 0.1214732 | +4.5% |
| 35 | 0.0050000 | 0.0818232 | +5.3% |
| **30** | **0.1236006** | **0.1246230** | **+0.007%** |

`N=30` — the run section 8 designates as the matched anti-selection-bias
control — is the single value whose incumbent is separation-healthy, and it is
the single value that barely improves. The control therefore does **not**
support a reading of "this search finds much better configurations". It
indicates that the large gains come mostly from not inheriting a constraint the
upstream search imposed.

The mathematical claim in section 2.4 remains exactly true as written: each new
literal configuration's proved lower bound does exceed the corresponding old
literal configuration's exact witness. Only the significance changes.

**Suggested fix.** State the upstream separation constraint in section 2.4, and
extend section 8's anti-selection-bias paragraph to say what the `N=30` control
actually shows rather than only that it was run.

### Finding 2 — the section 2.4 comparison column has inconsistent provenance and one understated value

**Severity: presentational, but it collides with the report's own section 9.**

The comparison values are 17-significant-digit renderings of the upstream
`rigorous_upper_witness`, but not by a single convention: rows `29,30,31,32`
reproduce a float64 round trip, while rows `33,34,35` reproduce exact-decimal
rounding. For `N=30` the printed `285.32674238354997` is strictly **below** the
true upstream witness `285.3267423835499873642554...`, and the trailing ellipsis
implies truncation of an exact decimal, which it is not.

No claim breaks — the margins run from 0.013 (`N=30`) to 14.4 (`N=33`) — but
section 9 explicitly warns reviewers not to compare against rounded values, and
the table then presents rounded values without saying so.

**Suggested fix.** Print exact rationals, or clearly-labeled truncations, and
use one convention for the whole column.

### Finding 3 — the `N=3` upper endpoint is not the tightest available, and the darkest point is misidentified for the published configuration

**Severity: minor. The stated interval is valid.**

Evaluating the *published truncated decimals* exactly:

```text
bottom edge midpoint (1/2, 0):  7.568389640029690214611494344582983...
corner (0, 0) and (1, 0):       7.568389640029690214292220397570000...
                                                    corner lower by 3.19e-19
```

Truncating the exact symmetric root to 19 significant digits breaks the
five-way level degeneracy, and the corner basin wins. Consequences:

- The tightest exact witness for the published configuration is the corner
  value, not the value at `(1/2,0)` quoted in section 2.1 and the README.
- The statement that numerical evaluation finds five equal-height active darkest
  points is true of the exact root but not of the configuration actually
  published; float64 cannot resolve the 3.19e-19 gap, which is why the numerical
  diagnostic reports them as equal.

**Suggested fix.** Quote the corner value as the upper endpoint, and note that
the degeneracy is exact only at the untruncated root.

### Finding 4 — section 8 omits how the large-`N` search was initialized

**Severity: provenance gap.**

Both run manifests record `initialization: incumbent-jitter`, `jitter: 0.001`,
and an `incumbent_root` pointing at the upstream submodule. The search is
therefore a local refinement of the public incumbents, not an independent
discovery. Optimal-assignment displacement from the upstream coordinates:

```text
N=29 max 0.0870   N=30 max 0.0032   N=31 max 0.0826   N=32 max 0.0834
N=33 max 0.0862   N=34 max 0.0952   N=35 max 0.0903
```

Section 8 describes the searcher's engineering in detail but never says this.
It is exactly the provenance a reviewer needs in order to read finding 1.

**Suggested fix.** Add the initialization mode, jitter magnitude, and incumbent
source to section 8.

### Finding 5 — "23 tests pass" holds only with PyTorch installed

**Severity: reproducibility friction.**

`tests/test_torch_engine.py` imports `torch` at module scope, so on a machine
without it `unittest discover` reports a hard `ERROR` and 20 passes, not a skip.
A reviewer following sections 5 and 6 — which the report itself says need no GPU
— hits this immediately, and `requirements.txt` pins `torch==2.9.0`, forcing a
large install for an audit that never uses it.

**Suggested fix.** Guard the module with `unittest.skipUnless`, and split
`requirements.txt` into exact-replay and search dependencies.

### Finding 6 — two gaps in the section 2.4 table

**Severity: minor.**

- There is no `N=3` row. `N=3` is the report's headline result, and section 2
  never states what it improves on; only the README carries the Friedman
  display value `7.507+`.
- The `N=5` row compares against a Friedman display value, which is exactly what
  section 9 tells reviewers not to do, without labeling the row as a different
  kind of comparison. The 3.4% margin makes this harmless in substance.

**Suggested fix.** Add the `N=3` row with its baseline, and mark the `N=5` row
as a web-display comparison because the public certificate library does not
cover `N=5`.

### Minor observations, no soundness impact

- `Interval.__pow__` in `square_riesz/exact_interval.py` squares by repeated
  multiplication, so an interval straddling zero yields `[-1,1]` instead of
  `[0,1]`. Still a valid enclosure, merely loose. The current KKT system never
  squares a zero-straddling interval, so nothing is affected today.
- `box_lower_bound` raises `ZeroDivisionError` on a degenerate zero-area box
  coinciding with a source. Unreachable from `certify_fixed_configuration`,
  whose bisection never produces a degenerate box.
- The componentwise certificates for `N=29,31,32,33,35` record markedly worse
  upper witnesses than their spectral twins (`N=35`: 444.68 against 347.20)
  because `--witness-x/--witness-y` were left at their defaults. Cosmetic —
  witness values never enter the lower-bound proof — but a reviewer diffing the
  two files for a given `N` will notice the discrepancy and wonder.

## 5. Claims confirmed, with nothing to report

- The two fixed-configuration lower methods do share `parse_decimal_points`,
  the branch-and-bound driver, and the direct bound. Section 2.1 and section 3
  item 6 already disclose this accurately; it is not an undisclosed weakness.
- The permutation reduction is sound. The 816 unordered triples over 16 covering
  cells cover every three-source configuration modulo source permutation,
  including coincident source cells, and the objective is symmetric in its
  sources.
- Witness admissibility is enforced correctly. Every leaf witness lies in the
  closed unit square and strictly outside all three source boxes, which is what
  makes the uniform bound finite and valid.
- The 95.9 MiB expanded JSON is correctly excluded from Git while the
  byte-identical zip is tracked, and `.gitattributes` pins `eol=lf` so the
  tracked hashes are stable across platforms.

## 6. Not checked

- Peer review or acceptance status of any record. Section 3 item 7 already
  disclaims this.
- Whether the `N=5` and `N=29..35` configurations are locally or globally
  optimal. Section 3 items 4 and 5 already disclaim this.
- Reproduction of the GPU search stage itself; no CUDA device was available to
  the audit. Only the persisted job structure and manifests were checked.
- The remaining twelve high-`N` replays. Section 6 recommends one per method
  before attempting all fourteen; that recommendation was followed.

## 7. Reproducing this audit

```bash
sha256sum data/certificates/n03_global_upper_7_58.zip
```

```bash
unzip -o data/certificates/n03_global_upper_7_58.zip -d data/certificates
```

```bash
python scripts/verify_global_upper_certificate.py data/certificates/n03_global_upper_7_58.json
```

```bash
python scripts/verify_global_upper_certificate_cleanroom.py data/certificates/n03_global_upper_7_58.json
```

The clean-room verifier depends only on the standard library and takes a single
positional certificate path. It prints `VERIFIED (clean-room)` together with the
exact maximum leaf bound, and raises on the first violation it encounters.
