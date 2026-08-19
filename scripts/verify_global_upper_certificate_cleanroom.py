"""Clean-room re-implementation of the N=3 global upper certificate verifier.

Written from the JSON schema only.  Deliberately different from the repository
verifier in: exact arithmetic representation (scaled integers, not Fraction),
coverage proof (Kraft equality + prefix-freeness via a trie, not recursion),
and root-cell / witness indexing derived from first principles.

Soundness argument being checked:
  I(a1,a2,a3) = min_{x in [0,1]^2} sum_i |x-a_i|^{-2}
  For a configuration box B = (B1,B2,B3) and any witness w in [0,1]^2 with
  w outside every B_i,
      I(X) <= sum_i |w-a_i|^{-2} <= sum_i dist(w,B_i)^{-2} =: U(w,B).
  If the B's cover ([0,1]^2)^3 modulo source permutation and every U(w,B) <= T,
  then P_3 = max_X I(X) <= T.
"""
import json, sys
from itertools import combinations_with_replacement

CERT = sys.argv[1]

with open(CERT, "r", encoding="utf-8") as fh:
    d = json.load(fh)

assert isinstance(d, dict)
assert d["n"] == 3, "not an N=3 certificate"
DIV = d["initial_divisions"]
GRID = d["witness_grid"]
assert type(DIV) is int and type(GRID) is int and DIV >= 1 and GRID >= 2
tgt_str = d["target"]
assert type(tgt_str) is str
# exact target as a fraction of integers, parsed by hand
if "." in tgt_str:
    ip, fp = tgt_str.split(".")
    assert ip.isdigit() and fp.isdigit()
    T_NUM, T_DEN = int(ip + fp), 10 ** len(fp)
else:
    assert tgt_str.isdigit()
    T_NUM, T_DEN = int(tgt_str), 1
assert T_NUM > 0
print(f"target parsed as {T_NUM}/{T_DEN}")

leaves = d["leaves"]
assert type(leaves) is list

EXPECTED_ROOTS = set(combinations_with_replacement(range(DIV * DIV), 3))
print(f"expected roots C({DIV*DIV}+2,3) = {len(EXPECTED_ROOTS)}")

# ---------------------------------------------------------------- geometry
# Everything is dyadic.  Represent a 1-D box [lo/2^s, hi/2^s] with integers,
# using scale s relative to the unit square.  Root cells have s = log2(DIV)
# only when DIV is a power of two, so instead keep an explicit denominator.

def root_boxes(root):
    """Return per-source [(xlo,xhi,den),(ylo,yhi,den)] as integers over DIV."""
    out = []
    for cell in root:
        assert 0 <= cell < DIV * DIV
        ix, iy = cell % DIV, cell // DIV
        out.append([[ix, ix + 1, DIV], [iy, iy + 1, DIV]])
    return out

def apply_path(root, path):
    b = root_boxes(root)
    for depth, ch in enumerate(path):
        c = depth % 6
        s, ax = c // 2, c % 2
        lo, hi, den = b[s][ax]
        lo, hi, den = 2 * lo, 2 * hi, 2 * den          # refine denominator
        mid = (lo + hi) // 2
        assert (lo + hi) % 2 == 0
        if ch == "0":
            hi = mid
        elif ch == "1":
            lo = mid
        else:
            raise AssertionError("bad path character")
        b[s][ax] = [lo, hi, den]
    return b

def dist1d_num(p_num, p_den, lo, hi, den):
    """Exact 1-D distance from p to [lo/den, hi/den] as (num, den) over p_den*den."""
    # common denominator p_den*den
    P = p_num * den
    L, H = lo * p_den, hi * p_den
    if P < L:
        return L - P
    if P > H:
        return P - H
    return 0

def leaf_ok(root, path, witness):
    """Return True iff U(w,B) <= target, exactly."""
    assert type(witness) is int and 0 <= witness < GRID * GRID
    wden = GRID - 1
    wx, wy = witness % GRID, witness // GRID          # both in [0, GRID-1]
    b = apply_path(root, path)
    # accumulate sum_i 1/d_i^2 as an exact fraction, integer numerator/denominator
    num, den = 0, 1
    for (xlo, xhi, xden), (ylo, yhi, yden) in b:
        # scale both axes to a common denominator with the witness
        dx = dist1d_num(wx, wden, xlo, xhi, xden)      # over wden*xden
        dy = dist1d_num(wy, wden, ylo, yhi, yden)      # over wden*yden
        # d^2 = (dx/(wden*xden))^2 + (dy/(wden*yden))^2
        ax, ay = wden * xden, wden * yden
        n2 = dx * dx * ay * ay + dy * dy * ax * ax
        d2 = (ax * ay) ** 2
        if n2 == 0:
            return False, None                          # witness touches a box
        num, den = num * n2 + den * d2, den * n2        # += d2/n2
    return (num * T_DEN <= T_NUM * den), (num, den)

# ---------------------------------------------------------------- main loop
by_root = {}
best = (0, 1)
best_f = -1.0
for i, leaf in enumerate(leaves):
    assert type(leaf) is dict
    r = leaf["root"]
    assert type(r) is list and len(r) == 3
    assert all(type(v) is int for v in r)
    root = tuple(r)
    assert root in EXPECTED_ROOTS, f"root {root} not a canonical multiset"
    path = leaf["path"]
    assert type(path) is str
    w = leaf["witness"]
    ok, val = leaf_ok(root, path, w)
    if not ok:
        raise SystemExit(f"LEAF {i} FAILS: root={root} path={path!r} witness={w} value={val}")
    f = val[0] / val[1]
    if f > best_f:
        best_f, best = f, val
    slot = by_root.setdefault(root, set())
    assert path not in slot, f"duplicate leaf path in root {root}"
    slot.add(path)
    if (i + 1) % 200000 == 0:
        print(f"  {i+1} leaves checked")

assert set(by_root) == EXPECTED_ROOTS, "root set mismatch"

# coverage: prefix-free + Kraft sum exactly 1  <=>  complete binary cover
for root, paths in by_root.items():
    L = max(map(len, paths))
    kraft = sum(1 << (L - len(p)) for p in paths)
    if kraft != (1 << L):
        raise SystemExit(f"root {root}: Kraft sum {kraft} != {1<<L} (incomplete or overlapping cover)")
    # prefix-freeness via trie insertion
    trie = {}
    for p in sorted(paths, key=len):
        node = trie
        for ch in p:
            assert "$" not in node, f"root {root}: path {p} extends a leaf"
            node = node.setdefault(ch, {})
        assert not node, f"root {root}: path {p} is a prefix of another leaf"
        node["$"] = True

assert len(leaves) == d["leaf_count"], "leaf_count metadata mismatch"

from decimal import Decimal, localcontext
with localcontext() as c:
    c.prec = 40
    dv = Decimal(best[0]) / Decimal(best[1])
print(json.dumps({
    "status": "VERIFIED (clean-room)",
    "target": f"{T_NUM}/{T_DEN}",
    "root_count": len(EXPECTED_ROOTS),
    "leaf_count": len(leaves),
    "max_leaf_bound_exact": f"{best[0]}/{best[1]}",
    "max_leaf_bound_decimal": str(dv),
}, indent=2))
