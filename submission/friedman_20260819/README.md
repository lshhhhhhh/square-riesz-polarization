# Friedman record-board submission package

Status: **the user reports that an email was sent externally**. The exact sent
message, public credit, attachment selection, and send timestamp were not
captured in this repository; see `SUBMISSION_STATUS.md`. Files here preserve
the reviewed pre-send package and must not be represented as an exact copy of
the transmitted email.

This directory packages nine finite-decimal configurations for Erich Friedman's
[Maximizing Minimum Light Intensity](https://erich-friedman.github.io/packing/light/)
board. The board and its displayed values were checked directly on 2026-08-19.
Friedman's current homepage exposes `erichfriedman68@gmail.com` as its “E-mail
Me” address.

The package follows the official
[Contributor Guidelines](https://erich-friedman.github.io/packing/submit.html):
the draft has a new descriptive subject, correctly named replacement GIFs,
intensities to five decimal places plus rigorous bounds, a method description,
and an explicit full-name placeholder. It is intended to be sent as a new email,
not as a reply to an earlier exchange.

## Proposed records

| `N` | Current board | Numerical `I` (5 d.p.) | Rigorous lower bound | Exact point-witness upper bound |
|---:|---:|---:|---:|---:|
| 3 | 7.507+ | **7.56839** | 7.56838963 | 7.5683896400296902142922... |
| 5 | 21.342+ | **22.06308** | 22.06 | 22.06308301603677... |
| 29 | 259.948+ | **282.85693** | 282.8 | 282.85692528612702... |
| 30 | 268.319+ | **285.34569** | 285.34 | 285.34568532988481... |
| 31 | 274.727+ | **305.29837** | 305.2 | 305.29836691152415... |
| 32 | 279.851+ | **317.20382** | 317.1 | 317.20381892782922... |
| 33 | 287.670+ | **330.59548** | 330.5 | 330.59547948010763... |
| 34 | 297.004+ | **337.90624** | 337.8 | 337.90623582303798... |
| 35 | 303.346+ | **347.19572** | 347.1 | 347.19572233129303... |

The numerical `I` is the continuous minimizer estimate used for the requested
five-decimal submission figure. The lower and upper endpoints have different
meanings. Each lower endpoint is
proved over the entire unit square by both exact-rational spectral and
componentwise branch-and-bound replays. Each upper endpoint is the exact value
at one published observer point, so it brackets the fixed configuration but is
not itself a lower-bound claim.

For `N=29..35`, the new lower bounds also exceed exact point-witness upper bounds
for the strongest pinned prior public literal configurations. For `N=3` and
`N=5`, only the displayed Friedman values are public, so the conservative claim
is “improves the displayed board value,” not “strictly defeats an unavailable
unrounded old configuration.”

## Attachments

- `records.csv`: board values, rigorous intervals, and exact source paths.
- `coordinates.csv`: all 232 source coordinates, indexed from 1 within each `N`.
- `metadata.json`: machine-readable scope and external status.
- `SHA256SUMS.txt`: hashes of all generated files, candidates, and certificates.
- `EMAIL_DRAFT.txt`: concise English draft; it deliberately contains a public
  attribution placeholder.
- `ATTACHMENT_CHECKLIST.txt`: the exact 11 primary attachments and final
  pre-send gates.
- `figures/3.gif`, `5.gif`, and `29.gif` through `35.gif`: replacement images
  named and sized exactly like the live files they replace. They use the required
  orientation and red-source/black-contour/blue-minimum color convention, with
  no text or outer border.

Regenerate the machine-built attachments with:

```bash
python scripts/build_friedman_figures.py
python scripts/build_friedman_submission.py
```

The GIF renderer additionally uses the pinned packages in
`requirements-submission.txt`. The differing live pixel heights (`225..231`)
are intentional: each new GIF matches the exact dimensions of the particular
image it replaces.

`EMAIL_DRAFT.txt` intentionally remains the reviewed template, including its
placeholders. The user handled transmission outside this project. This package
does not authorize a follow-up email or assert that Friedman accepted the
records.
