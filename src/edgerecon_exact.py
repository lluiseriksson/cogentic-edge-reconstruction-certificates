"""Pure-stdlib exact integer and rational arithmetic kernel for Paper 60:
Unconditional Edge Reconstruction of Simple Graphs of Linear Density (Cogentic 2026).

Strictly zero floating-point arithmetic: uses only Python arbitrary-precision `int`
and `fractions.Fraction`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations, permutations
from math import comb, factorial
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Strict Parsing (Zero-Float, Strict ASCII Digits)
# ---------------------------------------------------------------------------


def parse_exact_int(val: object, field: str) -> int:
    """Strictly parse an exact integer (rejecting bools, floats, non-ASCII digits)."""
    if isinstance(val, bool):
        raise ValueError(f"{field}: boolean not permitted where integer expected")
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            raise ValueError(f"{field}: empty integer string")
        sign = 1
        if s[0] == "-":
            sign = -1
            s = s[1:]
        elif s[0] == "+":
            s = s[1:]
        if not s or not (s.isascii() and s.isdigit()):
            raise ValueError(f"{field}: invalid exact integer string {val!r}")
        return sign * int(s)
    raise ValueError(f"{field}: expected exact int or integer string, got {type(val).__name__}")


def parse_exact_fraction(val: object, field: str) -> Fraction:
    """Strictly parse an exact rational 'num' or 'num/den' (rejecting floats/decimals)."""
    if isinstance(val, bool):
        raise ValueError(f"{field}: boolean not permitted where rational expected")
    if isinstance(val, int):
        return Fraction(val, 1)
    if isinstance(val, str):
        s = val.strip()
        if not s:
            raise ValueError(f"{field}: empty rational string")
        if "." in s or "e" in s or "E" in s:
            raise ValueError(f"{field}: decimal/scientific notation rejected in exact rational {val!r}")
        if "/" in s:
            parts = s.split("/")
            if len(parts) != 2:
                raise ValueError(f"{field}: malformed rational {val!r}")
            num = parse_exact_int(parts[0], f"{field}.num")
            den = parse_exact_int(parts[1], f"{field}.den")
            if den <= 0:
                raise ValueError(f"{field}: rational denominator must be positive, got {den}")
            return Fraction(num, den)
        return Fraction(parse_exact_int(s, field), 1)
    raise ValueError(f"{field}: expected exact rational string or int, got {type(val).__name__}")


def fmt_frac(q: Fraction) -> str:
    """Canonical exact rational string representation."""
    return f"{q.numerator}" if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


# ---------------------------------------------------------------------------
# Simple Graph Representation & Validation
# ---------------------------------------------------------------------------


def validate_simple_graph(n: int, edges_raw: Sequence[Sequence[object]], field: str) -> Tuple[int, List[Tuple[int, int]]]:
    """Validate simple undirected graph on vertices {0, ..., n-1}.

    Rejects self-loops, out-of-range vertices, malformed edges, and duplicate edges.
    Returns (n, sorted list of canonical edges (u, v) with u < v).
    """
    if n < 1:
        raise ValueError(f"{field}: vertex count n must be >= 1, got {n}")
    if not isinstance(edges_raw, list):
        raise ValueError(f"{field}: edges must be a list")
    seen: Set[Tuple[int, int]] = set()
    edges: List[Tuple[int, int]] = []
    for idx, e in enumerate(edges_raw):
        if not isinstance(e, (list, tuple)) or len(e) != 2:
            raise ValueError(f"{field}[{idx}]: edge must be a 2-element list")
        u = parse_exact_int(e[0], f"{field}[{idx}][0]")
        v = parse_exact_int(e[1], f"{field}[{idx}][1]")
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"{field}[{idx}]: vertex out of range [0, {n - 1}]: ({u}, {v})")
        if u == v:
            raise ValueError(f"{field}[{idx}]: self-loop ({u}, {u}) forbidden in simple graph")
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            raise ValueError(f"{field}[{idx}]: duplicate edge ({a}, {b}) forbidden in simple graph")
        seen.add((a, b))
        edges.append((a, b))
    edges.sort()
    return n, edges


def build_adj(n: int, edges: Sequence[Tuple[int, int]]) -> List[Set[int]]:
    """Build adjacency sets for a simple graph on {0, ..., n-1}."""
    adj: List[Set[int]] = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def degrees(adj: Sequence[Set[int]]) -> List[int]:
    return [len(nbrs) for nbrs in adj]


def degree_sequence(adj: Sequence[Set[int]]) -> List[int]:
    return sorted(len(nbrs) for nbrs in adj)


def connected_components(vertices: Iterable[int], adj: Sequence[Set[int]]) -> List[List[int]]:
    """Exact connected components of the induced subgraph on `vertices`."""
    vset = set(vertices)
    visited: Set[int] = set()
    comps: List[List[int]] = []
    for v in sorted(vset):
        if v in visited:
            continue
        stack = [v]
        visited.add(v)
        comp: List[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if nb in vset and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comp.sort()
        comps.append(comp)
    return comps


def is_connected(n: int, adj: Sequence[Set[int]]) -> bool:
    if n == 0:
        return True
    return len(connected_components(range(n), adj)) == 1


# ---------------------------------------------------------------------------
# Exact Graph Isomorphism, Automorphism Group, & Colored Canonical Labeling
# ---------------------------------------------------------------------------


def find_isomorphisms(
    n: int,
    adj1: Sequence[Set[int]],
    adj2: Sequence[Set[int]],
    colors1: Optional[Sequence[object]] = None,
    colors2: Optional[Sequence[object]] = None,
    limit: Optional[int] = None,
) -> List[Tuple[int, ...]]:
    """Exact backtracking search for all (or up to `limit`) color-preserving isomorphisms
    phi: V(G1) -> V(G2) with {u, v} in E1 <=> {phi(u), phi(v)} in E2.
    """
    if len(adj1) != n or len(adj2) != n:
        return []
    if colors1 is None:
        colors1 = [0] * n
    if colors2 is None:
        colors2 = [0] * n
    if Counter(colors1) != Counter(colors2):
        return []
    if sorted(len(s) for s in adj1) != sorted(len(s) for s in adj2):
        return []

    # Joint 1-WL color refinement using identical canonical signature mapping
    c1 = list(colors1)
    c2 = list(colors2)
    for _ in range(n + 1):
        sig1 = [(c1[v], len(adj1[v]), tuple(sorted(c1[u] for u in adj1[v]))) for v in range(n)]
        sig2 = [(c2[v], len(adj2[v]), tuple(sorted(c2[u] for u in adj2[v]))) for v in range(n)]
        if Counter(sig1) != Counter(sig2):
            return []
        all_sigs = sorted(set(sig1))
        idx_map = {s: i for i, s in enumerate(all_sigs)}
        nc1 = [idx_map[s] for s in sig1]
        nc2 = [idx_map[s] for s in sig2]
        if len(set(nc1)) == len(set(c1)):
            c1, c2 = nc1, nc2
            break
        c1, c2 = nc1, nc2

    # Group candidates in G2 by refined color
    buckets2: Dict[int, List[int]] = defaultdict(list)
    for v in range(n):
        buckets2[c2[v]].append(v)

    # Order vertices of G1 by smallest bucket size first, then highest degree
    order = sorted(range(n), key=lambda v: (len(buckets2[c1[v]]), -len(adj1[v]), v))

    mapping = [-1] * n
    used = [False] * n
    results: List[Tuple[int, ...]] = []

    def backtrack(pos: int) -> None:
        if limit is not None and len(results) >= limit:
            return
        if pos == n:
            results.append(tuple(mapping))
            return
        u = order[pos]
        u_nbrs = adj1[u]
        for w in buckets2[c1[u]]:
            if used[w]:
                continue
            w_nbrs = adj2[w]
            ok = True
            for prev_idx in range(pos):
                pu = order[prev_idx]
                pw = mapping[pu]
                if (pu in u_nbrs) != (pw in w_nbrs):
                    ok = False
                    break
            if not ok:
                continue
            mapping[u] = w
            used[w] = True
            backtrack(pos + 1)
            mapping[u] = -1
            used[w] = False
            if limit is not None and len(results) >= limit:
                return

    backtrack(0)
    return results


def are_isomorphic(
    n: int,
    adj1: Sequence[Set[int]],
    adj2: Sequence[Set[int]],
    colors1: Optional[Sequence[object]] = None,
    colors2: Optional[Sequence[object]] = None,
) -> bool:
    return len(find_isomorphisms(n, adj1, adj2, colors1, colors2, limit=1)) == 1


def automorphism_order(
    n: int,
    adj: Sequence[Set[int]],
    colors: Optional[Sequence[object]] = None,
) -> int:
    """Exact order of Aut(G) (or color-preserving Aut(G)) via exact backtracking."""
    return len(find_isomorphisms(n, adj, adj, colors, colors, limit=None))


def verify_isomorphism_map(
    n: int,
    adj1: Sequence[Set[int]],
    adj2: Sequence[Set[int]],
    phi: Sequence[int],
) -> bool:
    """Check whether `phi` is an exact graph isomorphism G1 -> G2."""
    if len(phi) != n:
        return False
    if set(phi) != set(range(n)):
        return False
    for u in range(n):
        for v in range(u + 1, n):
            if (v in adj1[u]) != (phi[v] in adj2[phi[u]]):
                return False
    return True


def decks_equal(
    cards1: Sequence[Tuple[int, List[Set[int]]]],
    cards2: Sequence[Tuple[int, List[Set[int]]]],
) -> bool:
    """Exact multiset isomorphism equality between two decks of graphs."""
    if len(cards1) != len(cards2):
        return False
    # Partition by isomorphism-invariant fingerprint (n, sorted degree sequence, edge count)
    def fp(card: Tuple[int, List[Set[int]]]) -> Tuple[int, int, Tuple[int, ...]]:
        nc, adj = card
        ds = tuple(sorted(len(s) for s in adj))
        return (nc, sum(ds) // 2, ds)

    g1: Dict[Tuple[int, int, Tuple[int, ...]], List[Tuple[int, List[Set[int]]]]] = defaultdict(list)
    g2: Dict[Tuple[int, int, Tuple[int, ...]], List[Tuple[int, List[Set[int]]]]] = defaultdict(list)
    for c in cards1:
        g1[fp(c)].append(c)
    for c in cards2:
        g2[fp(c)].append(c)
    if set(g1.keys()) != set(g2.keys()):
        return False
    for k in g1:
        l1 = g1[k]
        l2 = g2[k]
        if len(l1) != len(l2):
            return False
        # Group l1 and l2 into exact isomorphism classes
        classes1: List[Tuple[Tuple[int, List[Set[int]]], int]] = []
        for c in l1:
            placed = False
            for idx, (rep, cnt) in enumerate(classes1):
                if are_isomorphic(c[0], c[1], rep[1]):
                    classes1[idx] = (rep, cnt + 1)
                    placed = True
                    break
            if not placed:
                classes1.append((c, 1))
        matched = [0] * len(classes1)
        for c in l2:
            found = False
            for idx, (rep, _) in enumerate(classes1):
                if are_isomorphic(c[0], c[1], rep[1]):
                    matched[idx] += 1
                    found = True
                    break
            if not found:
                return False
        for idx, (_, cnt) in enumerate(classes1):
            if matched[idx] != cnt:
                return False
    return True


def edge_deck(n: int, edges: Sequence[Tuple[int, int]]) -> List[Tuple[int, List[Set[int]]]]:
    """Construct the exact edge-deck D_E(G) = {{G - e : e in E(G)}}."""
    cards: List[Tuple[int, List[Set[int]]]] = []
    for i in range(len(edges)):
        sub_edges = [e for j, e in enumerate(edges) if j != i]
        cards.append((n, build_adj(n, sub_edges)))
    return cards


def vertex_deck(n: int, edges: Sequence[Tuple[int, int]]) -> List[Tuple[int, List[Set[int]]]]:
    """Construct the exact vertex-deck D(G) = {{G - v : v in V(G)}}."""
    cards: List[Tuple[int, List[Set[int]]]] = []
    for v in range(n):
        remap = {u: (u if u < v else u - 1) for u in range(n) if u != v}
        sub_edges = [(remap[a], remap[b]) for a, b in edges if a != v and b != v]
        cards.append((n - 1, build_adj(n - 1, sub_edges)))
    return cards


# ---------------------------------------------------------------------------
# Exact Subgraph & Induced-Subgraph Counting (Kelly's Lemma)
# ---------------------------------------------------------------------------


STANDARD_PATTERNS: Dict[str, Tuple[int, List[Tuple[int, int]]]] = {
    "K2": (2, [(0, 1)]),
    "P3": (3, [(0, 1), (1, 2)]),
    "2K2": (4, [(0, 1), (2, 3)]),
    "K3": (3, [(0, 1), (1, 2), (0, 2)]),
    "K1_3": (4, [(0, 1), (0, 2), (0, 3)]),
    "P4": (4, [(0, 1), (1, 2), (2, 3)]),
    "C4": (4, [(0, 1), (1, 2), (2, 3), (0, 3)]),
}


def count_subgraphs(n: int, edges: Sequence[Tuple[int, int]], pat_name: str) -> int:
    """Exact number of edge-subsets E0 subseteq E(G) whose spanning graph (ignoring isolates)
    is isomorphic to the isolate-free pattern `pat_name`.
    """
    if pat_name not in STANDARD_PATTERNS:
        raise ValueError(f"Unknown subgraph pattern {pat_name!r}")
    pn, pedges = STANDARD_PATTERNS[pat_name]
    pm = len(pedges)
    if pm > len(edges):
        return 0
    padj = build_adj(pn, pedges)
    count = 0
    for chosen in combinations(edges, pm):
        verts = sorted({u for e in chosen for u in e})
        if len(verts) != pn:
            continue
        vmap = {v: i for i, v in enumerate(verts)}
        cadj = build_adj(pn, [(vmap[a], vmap[b]) for a, b in chosen])
        if are_isomorphic(pn, cadj, padj):
            count += 1
    return count


def count_induced_subgraphs(n: int, adj: Sequence[Set[int]], pat_name: str) -> int:
    """Exact ind(F, G) = |{W subseteq V(G) : G[W] ~= F}|."""
    if pat_name not in STANDARD_PATTERNS:
        raise ValueError(f"Unknown subgraph pattern {pat_name!r}")
    pn, pedges = STANDARD_PATTERNS[pat_name]
    if pn > n:
        return 0
    padj = build_adj(pn, pedges)
    count = 0
    for verts in combinations(range(n), pn):
        vmap = {v: i for i, v in enumerate(verts)}
        sub_edges = [
            (vmap[a], vmap[b])
            for idx_a, a in enumerate(verts)
            for b in verts[idx_a + 1 :]
            if b in adj[a]
        ]
        if len(sub_edges) != len(pedges):
            continue
        if are_isomorphic(pn, build_adj(pn, sub_edges), padj):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Section 3: Lexicographic Degree-Ordered Kruskal MST & Multi-Scale Rigidity
# ---------------------------------------------------------------------------


def kruskal_degree_ordered_mst(
    n: int,
    edges: Sequence[Tuple[int, int]],
    vertex_order: Sequence[int],
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Compute the UNIQUE lexicographic degree-ordered Kruskal MST (T, <_V) (Sec. 3).

    `vertex_order` lists all vertices in strictly increasing <_V order:
    vertex_order[0] <_V vertex_order[1] <_V ... <_V vertex_order[n-1] = r0.
    Requires d_G(x) < d_G(y) => rank[x] < rank[y].
    Each edge e = {x, y} with rank[y] < rank[x] is ordered lexicographically by (rank[x], rank[y]).
    Returns (sorted MST edge list, rank array).
    """
    adj = build_adj(n, edges)
    degs = degrees(adj)
    if len(vertex_order) != n or set(vertex_order) != set(range(n)):
        raise ValueError("vertex_order must be a permutation of {0, ..., n-1}")
    rank = [0] * n
    for pos, v in enumerate(vertex_order):
        rank[v] = pos
    # Check degree-monotonicity of <_V
    for i in range(n - 1):
        if degs[vertex_order[i]] > degs[vertex_order[i + 1]]:
            raise ValueError(
                f"vertex_order violates degree monotonicity at index {i}: "
                f"d({vertex_order[i]})={degs[vertex_order[i]]} > d({vertex_order[i+1]})={degs[vertex_order[i+1]]}"
            )
    if not is_connected(n, adj):
        raise ValueError("Graph must be connected for Kruskal MST")

    def edge_key(e: Tuple[int, int]) -> Tuple[int, int]:
        u, v = e
        if rank[v] < rank[u]:
            return (rank[u], rank[v])
        return (rank[v], rank[u])

    ordered_edges = sorted(edges, key=edge_key)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mst: List[Tuple[int, int]] = []
    for u, v in ordered_edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            mst.append((u, v) if u < v else (v, u))
            if len(mst) == n - 1:
                break
    mst.sort()
    return mst, rank


def phi_delta_functional(adj: Sequence[Set[int]], delta: int) -> int:
    """Lemma 6.1(1) piecewise-linear degree functional:
    Phi_Delta(F) = sum_{x in V(F)} max(0, d_F(x) - Delta + 1).
    """
    return sum(max(0, len(nbrs) - delta + 1) for nbrs in adj)


def analyze_kruskal_scale(
    n: int,
    adj: Sequence[Set[int]],
    mst_edges: Sequence[Tuple[int, int]],
    rank: Sequence[int],
    delta_k: int,
    delta_next: int,
) -> Dict[str, object]:
    """Compute exact Lemma 3.1(1) multi-scale diagnostics at threshold Delta_k."""
    degs = degrees(adj)
    Hk = {v for v in range(n) if degs[v] >= delta_k}
    Bk = {v for v in range(n) if degs[v] < delta_k}
    Uk = {v for v in Hk if degs[v] < delta_next}

    # Spanning forest edges of T with weight < delta_k (both endpoints in Bk)
    low_t_edges = [(u, v) for (u, v) in mst_edges if max(degs[u], degs[v]) < delta_k]
    comps = connected_components(Bk, adj)
    t_adj = build_adj(n, mst_edges)

    # Verify T[V(C)] is a spanning tree of C (Lemma 3.1(1a))
    low_t_adj = build_adj(n, low_t_edges)
    low_t_comps = connected_components(Bk, low_t_adj)
    spanning_forest_ok = (low_t_comps == comps)

    bridge_comps: List[List[int]] = []
    non_bridge_pk: List[Tuple[List[int], int]] = []  # (comp vertices, canonical parent u(C))
    canonical_parent_all_ok = True

    for comp in comps:
        cset = set(comp)
        exit_t_edges = [
            (u, v)
            for u in comp
            for v in t_adj[u]
            if v not in cset
        ]
        deg_t_c = len(exit_t_edges)
        if deg_t_c >= 2:
            bridge_comps.append(comp)
        elif deg_t_c == 1:
            u_exit = exit_t_edges[0][1]
            # Canonical parent u(C) = min_{<_V} (N_G(C) cap H_k)
            ext_nbrs_hk = {nb for u in comp for nb in adj[u] if nb in Hk}
            if not ext_nbrs_hk:
                canonical_parent_all_ok = False
                continue
            u_canon = min(ext_nbrs_hk, key=lambda x: rank[x])
            if u_exit != u_canon:
                canonical_parent_all_ok = False
            if u_canon in Uk:
                non_bridge_pk.append((comp, u_canon))
        else:
            # Connected graph with Hk != empty must have deg_T(C) >= 1
            if Hk:
                canonical_parent_all_ok = False

    # Partition Pk into exact Hk-labeled isomorphism classes F_k
    classes: List[Tuple[List[int], int, int]] = []  # (rep comp, parent u, multiplicity)
    for comp, u_canon in non_bridge_pk:
        k_sz = len(comp)
        vmap = {v: i for i, v in enumerate(comp)}
        cedges = [
            (vmap[a], vmap[b])
            for idx_a, a in enumerate(comp)
            for b in comp[idx_a + 1 :]
            if b in adj[a]
        ]
        cadj = build_adj(k_sz, cedges)
        clabels = [tuple(sorted(nb for nb in adj[v] if nb in Hk)) for v in comp]

        matched = False
        for idx, (rep_comp, rep_u, cnt) in enumerate(classes):
            if len(rep_comp) != k_sz or rep_u != u_canon:
                continue
            rmap = {v: i for i, v in enumerate(rep_comp)}
            redges = [
                (rmap[a], rmap[b])
                for idx_a, a in enumerate(rep_comp)
                for b in rep_comp[idx_a + 1 :]
                if b in adj[a]
            ]
            radj = build_adj(k_sz, redges)
            rlabels = [tuple(sorted(nb for nb in adj[v] if nb in Hk)) for v in rep_comp]
            if are_isomorphic(k_sz, cadj, radj, clabels, rlabels):
                classes[idx] = (rep_comp, rep_u, cnt + 1)
                matched = True
                break
        if not matched:
            classes.append((comp, u_canon, 1))

    mults = sorted([cnt for _, _, cnt in classes], reverse=True)
    aut_factor = 1
    for w in mults:
        aut_factor *= factorial(w)

    # Core tree edges: |H_k| + |B_{Delta_k}| - 1 when H_k nonempty
    core_edges = (len(Hk) + len(bridge_comps) - 1) if Hk else 0

    return {
        "delta_k": delta_k,
        "delta_next": delta_next,
        "H_k_size": len(Hk),
        "B_k_size": len(Bk),
        "U_k_size": len(Uk),
        "num_components_Bk": len(comps),
        "spanning_forest_ok": spanning_forest_ok,
        "bridge_count": len(bridge_comps),
        "bridge_bound_ok": (len(bridge_comps) <= len(Hk) - 1) if Hk else (len(bridge_comps) == 0),
        "core_tree_edges": core_edges,
        "core_tree_edge_bound_ok": (core_edges <= 2 * len(Hk) - 2) if Hk else True,
        "canonical_parent_all_ok": canonical_parent_all_ok,
        "non_bridge_Pk_count": len(non_bridge_pk),
        "class_multiplicities": mults,
        "aut_subgroup_factor": aut_factor,
    }


# ---------------------------------------------------------------------------
# Section 4: Exact Shift-Invariant Polynomials, Charlier Coefficients, & Band Isolation
# ---------------------------------------------------------------------------


def parse_sparse_poly(
    terms_raw: Sequence[Dict[str, object]],
    num_vars: int,
    field: str,
) -> Dict[Tuple[int, ...], Fraction]:
    """Parse sparse multivariate rational polynomial with duplicate-monomial & zero checks."""
    if num_vars < 1:
        raise ValueError(f"{field}: num_vars must be >= 1")
    if not isinstance(terms_raw, list) or not terms_raw:
        raise ValueError(f"{field}: polynomial terms must be a non-empty list")
    poly: Dict[Tuple[int, ...], Fraction] = {}
    for idx, t in enumerate(terms_raw):
        if not isinstance(t, dict) or set(t.keys()) != {"exp", "coeff"}:
            raise ValueError(f"{field}[{idx}]: term must have exact keys {{'exp', 'coeff'}}")
        exp_raw = t["exp"]
        if not isinstance(exp_raw, list) or len(exp_raw) != num_vars:
            raise ValueError(f"{field}[{idx}].exp: must be list of length {num_vars}")
        exp = tuple(parse_exact_int(x, f"{field}[{idx}].exp[{j}]") for j, x in enumerate(exp_raw))
        if any(x < 0 for x in exp):
            raise ValueError(f"{field}[{idx}].exp: negative exponent forbidden: {exp}")
        coeff = parse_exact_fraction(t["coeff"], f"{field}[{idx}].coeff")
        if coeff == 0:
            raise ValueError(f"{field}[{idx}].coeff: zero coefficient forbidden in canonical support")
        if exp in poly:
            raise ValueError(f"{field}[{idx}]: duplicate exponent vector {exp}")
        poly[exp] = coeff
    return poly


def poly_directional_derivative(poly: Dict[Tuple[int, ...], Fraction]) -> Dict[Tuple[int, ...], Fraction]:
    """Compute D_I(P) = sum_{i=1}^M dP/dy_i exactly over Q."""
    deriv: Dict[Tuple[int, ...], Fraction] = defaultdict(Fraction)
    for exp, coeff in poly.items():
        for i, ei in enumerate(exp):
            if ei > 0:
                new_exp = exp[:i] + (ei - 1,) + exp[i + 1 :]
                deriv[new_exp] += coeff * ei
    return {k: v for k, v in deriv.items() if v != 0}


def poly_l1_norm(poly: Dict[Tuple[int, ...], Fraction]) -> Fraction:
    return sum((abs(c) for c in poly.values()), Fraction(0, 1))


def charlier_scaled_exact(d: int, k: int, r: int, lam: Fraction) -> Fraction:
    """Exact rational scaled Charlier coefficient (Sec. 4.1):
    I_{d, k}(r; lam) = (q_d(k) / d!) * r^(d - k),
    where q_d(k) / d! = sum_{j=0}^d ((-lam)^(d - j) / (d - j)!) * C(k, j).
    """
    if d < 1 or k < 0 or r < 1 or lam <= 0:
        raise ValueError(f"Invalid Charlier parameters: d={d}, k={k}, r={r}, lam={lam}")
    qd_over_dfact = Fraction(0, 1)
    for j in range(d + 1):
        if j > k:
            continue
        term = ((-lam) ** (d - j)) / factorial(d - j) * comb(k, j)
        qd_over_dfact += term
    scale = Fraction(r, 1) ** (d - k) if d >= k else Fraction(1, r ** (k - d))
    return qd_over_dfact * scale


def lemma43_band_isolation(
    poly: Dict[Tuple[int, ...], Fraction],
    v: Tuple[int, ...],
    t0: int,
    t1: int,
) -> Dict[str, object]:
    """Exact Lemma 4.3 multiplicity-band isolation operator:
    I_lo = {i : v_i < t0}, I_mid = {i : t0 <= v_i <= t1}, I_hi = {i : v_i > t1}.
    Computes w_i = v_i (i in I_lo), w_i = t0 (i in I_mid cup I_hi),
    H_w(P) = (1/w!) d^w P, F = (1/C_const) H_w(P), F_mid = [y_{I_hi}^{gamma_hi}] F.
    """
    M = len(v)
    I_lo = [i for i in range(M) if v[i] < t0]
    I_mid = [i for i in range(M) if t0 <= v[i] <= t1]
    I_hi = [i for i in range(M) if v[i] > t1]

    w = tuple(v[i] if i in I_lo else t0 for i in range(M))
    gamma_hi = tuple(v[i] - t0 for i in I_hi)

    # Constant C_const = prod_{i in I_mid cup I_hi} C(v_i, t0)
    c_const = 1
    for i in I_mid + I_hi:
        c_const *= comb(v[i], t0)
    if c_const == 0:
        raise ValueError("C_const vanished in Lemma 4.3 band isolation")

    # Apply H_w(P): survives iff u_i >= w_i for all i, multiplied by prod C(u_i, w_i)
    hw: Dict[Tuple[int, ...], Fraction] = {}
    for u, coeff in poly.items():
        if all(u[i] >= w[i] for i in range(M)):
            mult = 1
            for i in range(M):
                mult *= comb(u[i], w[i])
            new_u = tuple(u[i] - w[i] for i in range(M))
            hw[new_u] = coeff * mult

    # Check independence from I_lo
    lo_independent = all(all(u[i] == 0 for i in I_lo) for u in hw)

    # Divide by C_const and extract [y_{I_hi}^{gamma_hi}]
    f_mid: Dict[Tuple[int, ...], Fraction] = {}
    for u, coeff in hw.items():
        if all(u[i] == 0 for i in I_lo) and tuple(u[i] for i in I_hi) == gamma_hi:
            z_mid = tuple(u[i] for i in I_mid)
            f_mid[z_mid] = f_mid.get(z_mid, Fraction(0, 1)) + coeff / c_const
    f_mid = {k: val for k, val in f_mid.items() if val != 0}

    target_mid = tuple(v[i] - t0 for i in I_mid)
    deriv_mid = poly_directional_derivative(f_mid) if I_mid else {}

    return {
        "I_lo": I_lo,
        "I_mid": I_mid,
        "I_hi": I_hi,
        "C_const": c_const,
        "lo_independent": lo_independent,
        "f_mid": f_mid,
        "target_mid": target_mid,
        "target_mid_coeff": f_mid.get(target_mid, Fraction(0, 1)),
        "f_mid_l1": poly_l1_norm(f_mid),
        "f_mid_degree": sum(target_mid),
        "f_mid_shift_invariant": (len(deriv_mid) == 0),
    }


# ---------------------------------------------------------------------------
# Section 5: Bounded-Degree Stabilizers & Single-Bridge Automorphism Rigidity
# ---------------------------------------------------------------------------


def prime_factors(n: int) -> List[int]:
    """Exact trial division prime factorization for positive integer n."""
    if n < 1:
        raise ValueError(f"prime_factors requires n >= 1, got {n}")
    primes: List[int] = []
    d = 2
    x = n
    while d * d <= x:
        if x % d == 0:
            primes.append(d)
            while x % d == 0:
                x //= d
        d += 1 if d == 2 else 2
    if x > 1:
        primes.append(x)
    return primes


def p_adic_val(n: int, p: int) -> int:
    if n <= 0 or p < 2:
        raise ValueError("Invalid arguments for p_adic_val")
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def legendre_ord_fact(R: int, p: int) -> int:
    """Exact Legendre p-adic valuation ord_p(R!)."""
    total = 0
    cur = R // p
    while cur > 0:
        total += cur
        cur //= p
    return total


def compute_bridge_restriction_group(
    n: int,
    adj: Sequence[Set[int]],
    delta: int,
    comp_D: Sequence[int],
    v0: int,
    fixed_S: Sequence[int],
) -> Dict[str, object]:
    """Compute exact Lemma 5.1(2) group G_{v0, S cap W_D}(D, W_D) and its restriction
    Q_D = {theta|_{W_D \\ S} : (theta, psi) in G_{v0, S cap W_D}(D, W_D)}.
    """
    degs = degrees(adj)
    H_delta = {v for v in range(n) if degs[v] >= delta}
    B_delta = {v for v in range(n) if degs[v] < delta}
    d_set = set(comp_D)
    if not d_set or not d_set.issubset(B_delta):
        raise ValueError("comp_D must be a non-empty subset of V_{<Delta}(G)")
    # Verify comp_D is an exact connected component of G[B_delta]
    all_comps = connected_components(B_delta, adj)
    if sorted(comp_D) not in all_comps:
        raise ValueError("comp_D is not a connected component of G[V_{<Delta}(G)]")
    if v0 not in d_set:
        raise ValueError("Root vertex v0 must lie in comp_D")

    W_D = sorted({nb for v in comp_D for nb in adj[v] if nb in H_delta})
    S_set = set(fixed_S)
    if not S_set.issubset(H_delta):
        raise ValueError("Fixed set S must be a subset of V_{>=Delta}(G)")
    S_cap_W = sorted(a for a in W_D if a in S_set)
    W_free = sorted(a for a in W_D if a not in S_set)

    # Build induced subgraph D
    d_list = sorted(comp_D)
    d_len = len(d_list)
    d_map = {v: i for i, v in enumerate(d_list)}
    d_edges = [
        (d_map[a], d_map[b])
        for idx_a, a in enumerate(d_list)
        for b in d_list[idx_a + 1 :]
        if b in adj[a]
    ]
    d_adj = build_adj(d_len, d_edges)

    # Automorphisms psi in Aut(D) fixing v0 and preserving neighborhood in S cap W_D
    # and preserving total degree into W_free!
    init_colors = [
        (
            1 if v == v0 else 0,
            tuple(sorted(a for a in S_cap_W if a in adj[v])),
            sum(1 for a in W_free if a in adj[v]),
        )
        for v in d_list
    ]
    psi_list = find_isomorphisms(d_len, d_adj, d_adj, init_colors, init_colors, limit=None)

    # For each psi, find all compatible bijections theta on W_free
    # Condition: for all a in W_free, v in D: a in N(v) <=> theta(a) in N(psi(v))
    # Define signature of a in W_free under identity: sig_orig(a) = frozenset(v in D : a in N(v))
    # Under psi, theta(a) must satisfy sig_orig(theta(a)) == psi(sig_orig(a)).
    orig_by_sig: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    sig_of_a: Dict[int, Tuple[int, ...]] = {}
    for a in W_free:
        sig = tuple(i for i, v in enumerate(d_list) if a in adj[v])
        sig_of_a[a] = sig
        orig_by_sig[sig].append(a)

    q_elements: Set[Tuple[int, ...]] = set()
    group_order = 0

    for psi in psi_list:
        # Check whether every a in W_free has image signature present with equal multiplicity
        img_sigs: Dict[int, Tuple[int, ...]] = {}
        ok = True
        for a in W_free:
            mapped_sig = tuple(sorted(psi[i] for i in sig_of_a[a]))
            if mapped_sig not in orig_by_sig:
                ok = False
                break
            img_sigs[a] = mapped_sig
        if not ok:
            continue
        # Group W_free by their target signature
        by_target: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
        for a in W_free:
            by_target[img_sigs[a]].append(a)
        if any(len(by_target[s]) != len(orig_by_sig[s]) for s in by_target):
            continue
        # Enforce explicit cap on |W_free| before factorial permutation enumeration
        if len(W_free) > 8:
            raise ValueError(f"|W_free|={len(W_free)} exceeds exact permutation enumeration cap 8")
        bucket_keys = sorted(by_target.keys())
        bucket_choices = [list(permutations(orig_by_sig[s])) for s in bucket_keys]

        def enum_theta(b_idx: int, cur_map: Dict[int, int]) -> None:
            nonlocal group_order
            if b_idx == len(bucket_keys):
                q_elements.add(tuple(cur_map[a] for a in W_free))
                group_order += 1
                return
            src_list = by_target[bucket_keys[b_idx]]
            for perm_dst in bucket_choices[b_idx]:
                for s_a, d_a in zip(src_list, perm_dst):
                    cur_map[s_a] = d_a
                enum_theta(b_idx + 1, cur_map)

        enum_theta(0, {})

    q_order = len(q_elements)
    R = len(W_free)
    primes_q = prime_factors(q_order) if q_order > 1 else []
    all_primes_lt_delta = all(p < delta for p in primes_q)
    bound_16delta_R = (16 * delta) ** R
    legendre_checks = {
        str(p): {
            "ord_p_Q": p_adic_val(q_order, p),
            "ord_p_R_fact": legendre_ord_fact(R, p),
            "r_over_p_minus_1_floor": R // (p - 1),
        }
        for p in primes_q
    }

    return {
        "W_D": W_D,
        "S_cap_W_D": S_cap_W,
        "W_free": W_free,
        "R": R,
        "psi_candidate_count": len(psi_list),
        "full_group_order": group_order,
        "Q_D_order": q_order,
        "Q_D_prime_divisors": primes_q,
        "all_primes_lt_delta": all_primes_lt_delta,
        "bound_16delta_R": bound_16delta_R,
        "within_16delta_R_bound": q_order <= bound_16delta_R,
        "legendre_checks": legendre_checks,
    }
