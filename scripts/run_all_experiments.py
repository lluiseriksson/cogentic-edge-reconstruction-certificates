"""Generate all 16 canonical exact certificates and 4 exact numerical datasets for Paper 60:
Unconditional Edge Reconstruction of Simple Graphs of Linear Density (Cogentic 2026).
"""

from __future__ import annotations

import json
import sys
import time
from fractions import Fraction
from math import comb, factorial, log2
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.edgerecon_certifier import SCHEMA_VERSION, verify_certificate_dict
from src.edgerecon_exact import (
    analyze_kruskal_scale,
    are_isomorphic,
    automorphism_order,
    build_adj,
    charlier_scaled_exact,
    compute_bridge_restriction_group,
    count_induced_subgraphs,
    count_subgraphs,
    decks_equal,
    degrees,
    edge_deck,
    fmt_frac,
    kruskal_degree_ordered_mst,
    lemma43_band_isolation,
    poly_directional_derivative,
    poly_l1_norm,
    vertex_deck,
)


CERT_DIR = ROOT / "certificates"
RES_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Certificate Builders (Exact, Zero-Float)
# ---------------------------------------------------------------------------


def make_mode1_cert(
    cert_id: str,
    description: str,
    n: int,
    edges_G: List[Tuple[int, int]],
    edges_H: List[Tuple[int, int]],
    verdict: str,
    kelly_patterns: List[str],
    isomorphism_phi: List[int] | None = None,
) -> Dict[str, Any]:
    adj_G = build_adj(n, edges_G)
    adj_H = build_adj(n, edges_H)
    ed_eq = decks_equal(edge_deck(n, edges_G), edge_deck(n, edges_H))
    vd_eq = decks_equal(vertex_deck(n, edges_G), vertex_deck(n, edges_H))
    aut_G = automorphism_order(n, adj_G)
    aut_H = automorphism_order(n, adj_H)
    ke = {p: count_subgraphs(n, edges_G, p) for p in kelly_patterns}
    kv = {p: count_induced_subgraphs(n, adj_G, p) for p in kelly_patterns}
    muller = (len(edges_G) >= 1) and ((1 << (len(edges_G) - 1)) > factorial(n))

    payload: Dict[str, Any] = {
        "n": n,
        "edges_G": [[u, v] for u, v in edges_G],
        "edges_H": [[u, v] for u, v in edges_H],
        "verdict": verdict,
        "claimed_edge_decks_equal": ed_eq,
        "claimed_vertex_decks_equal": vd_eq,
        "claimed_aut_G": aut_G,
        "claimed_aut_H": aut_H,
        "kelly_patterns": kelly_patterns,
        "claimed_kelly_edge_counts_G": ke,
        "claimed_kelly_vertex_counts_G": kv,
        "claimed_muller_holds": muller,
    }
    if isomorphism_phi is not None:
        payload["isomorphism_phi"] = isomorphism_phi
    return {
        "schema_version": SCHEMA_VERSION,
        "cert_id": cert_id,
        "mode": "edge_deck_hypomorphism_and_kelly",
        "description": description,
        "payload": payload,
    }


def make_mode2_cert(
    cert_id: str,
    description: str,
    n: int,
    edges: List[Tuple[int, int]],
    thresholds: List[int],
    phi_delta_scale: int,
    include_exact_aut: bool = False,
) -> Dict[str, Any]:
    adj = build_adj(n, edges)
    degs = degrees(adj)
    # Canonical degree-monotone vertex order <_V
    vertex_order = sorted(range(n), key=lambda v: (degs[v], v))
    mst_edges, rank = kruskal_degree_ordered_mst(n, edges, vertex_order)
    scale_summaries: List[Dict[str, Any]] = []
    aut_prod = 1
    for k in range(len(thresholds) - 1):
        res = analyze_kruskal_scale(n, adj, mst_edges, rank, thresholds[k], thresholds[k + 1])
        scale_summaries.append(
            {
                "delta_k": res["delta_k"],
                "H_k_size": res["H_k_size"],
                "bridge_count": res["bridge_count"],
                "core_tree_edges": res["core_tree_edges"],
                "class_multiplicities": res["class_multiplicities"],
                "aut_subgroup_factor": res["aut_subgroup_factor"],
            }
        )
        aut_prod *= int(res["aut_subgroup_factor"])

    payload: Dict[str, Any] = {
        "n": n,
        "edges": [[u, v] for u, v in sorted(edges)],
        "vertex_order": vertex_order,
        "thresholds": thresholds,
        "claimed_mst_edges": [[u, v] for u, v in mst_edges],
        "claimed_scale_summaries": scale_summaries,
        "claimed_aut_product_bound": aut_prod,
        "claimed_phi_delta_check_scale": phi_delta_scale,
    }
    if include_exact_aut:
        payload["claimed_exact_aut_order"] = automorphism_order(n, adj)
    return {
        "schema_version": SCHEMA_VERSION,
        "cert_id": cert_id,
        "mode": "kruskal_multiscale_rigidity_cert",
        "description": description,
        "payload": payload,
    }


def poly_mul(
    p1: Dict[Tuple[int, ...], Fraction],
    p2: Dict[Tuple[int, ...], Fraction],
) -> Dict[Tuple[int, ...], Fraction]:
    out: Dict[Tuple[int, ...], Fraction] = {}
    for e1, c1 in p1.items():
        for e2, c2 in p2.items():
            e = tuple(a + b for a, b in zip(e1, e2))
            out[e] = out.get(e, Fraction(0, 1)) + c1 * c2
    return {k: v for k, v in out.items() if v != 0}


def diff_linear(M: int, i: int, j: int) -> Dict[Tuple[int, ...], Fraction]:
    """(y_i - y_j) in Q[y_0, ..., y_{M-1}]."""
    ei = tuple(1 if k == i else 0 for k in range(M))
    ej = tuple(1 if k == j else 0 for k in range(M))
    return {ei: Fraction(1, 1), ej: Fraction(-1, 1)}


def poly_prod(factors: List[Dict[Tuple[int, ...], Fraction]], M: int) -> Dict[Tuple[int, ...], Fraction]:
    cur: Dict[Tuple[int, ...], Fraction] = {(0,) * M: Fraction(1, 1)}
    for f in factors:
        cur = poly_mul(cur, f)
    return cur


def make_mode3_cert(
    cert_id: str,
    description: str,
    M: int,
    poly: Dict[Tuple[int, ...], Fraction],
    target_v: Tuple[int, ...],
    t_max: int,
    charlier_d: int,
    charlier_r: int,
    charlier_kmax: int,
    band_bounds: Tuple[int, int] | None = None,
) -> Dict[str, Any]:
    N0 = sum(target_v)
    assert len(poly_directional_derivative(poly)) == 0, "Polynomial must be shift-invariant"
    cv = poly[target_v]
    l1 = poly_l1_norm(poly)
    ratio = l1 / abs(cv)

    lam = Fraction(1, 2 * charlier_r)
    ub = Fraction(1, 1) - Fraction(11 * charlier_d, 24 * charlier_r)
    max_abs = max(abs(charlier_scaled_exact(charlier_d, k, charlier_r, lam)) for k in range(charlier_kmax + 1))

    terms = [
        {"exp": list(exp), "coeff": fmt_frac(coeff)}
        for exp, coeff in sorted(poly.items())
    ]
    payload: Dict[str, Any] = {
        "num_vars": M,
        "degree_N0": N0,
        "terms": terms,
        "target_v": list(target_v),
        "t_max": t_max,
        "claimed_l1_norm": fmt_frac(l1),
        "claimed_target_coeff": fmt_frac(cv),
        "claimed_ratio_l1_over_cv": fmt_frac(ratio),
        "charlier_check": {
            "d": charlier_d,
            "r": charlier_r,
            "k_max": charlier_kmax,
            "claimed_max_abs_I": fmt_frac(max_abs),
            "claimed_upper_bound": fmt_frac(ub),
        },
    }
    if band_bounds is not None:
        t0, t1 = band_bounds
        bres = lemma43_band_isolation(poly, target_v, t0, t1)
        payload["band_isolation"] = {
            "t0": t0,
            "t1": t1,
            "claimed_C_const": bres["C_const"],
            "claimed_f_mid_degree": bres["f_mid_degree"],
            "claimed_f_mid_l1": fmt_frac(bres["f_mid_l1"]),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "cert_id": cert_id,
        "mode": "shift_invariant_charlier_band_cert",
        "description": description,
        "payload": payload,
    }


def make_mode4_cert(
    cert_id: str,
    description: str,
    n: int,
    edges: List[Tuple[int, int]],
    delta: int,
    comp_D: List[int],
    v0: int,
    fixed_S: List[int],
    n0: int,
    m0: int,
    pillar3_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    adj = build_adj(n, edges)
    bres = compute_bridge_restriction_group(n, adj, delta, comp_D, v0, fixed_S)
    avg_d = Fraction(2 * m0, n0)
    if n0 <= 2000:
        regime = "VACUOUS_SIMPLE_GRAPH_CAP"
    elif m0 >= 1000 * n0:
        regime = "ACTIVE_LINEAR_DENSITY"
    else:
        regime = "BELOW_1000N_THRESHOLD"
    bl = n0.bit_length()
    if m0 == 0:
        mul_exceeds = False
    elif m0 - 1 > n0 * bl:
        mul_exceeds = True
    elif m0 - 1 < n0 * (bl - 1):
        mul_exceeds = False
    else:
        mul_exceeds = (1 << (m0 - 1)) > (n0 ** n0)

    payload: Dict[str, Any] = {
        "n": n,
        "edges": [[u, v] for u, v in sorted(edges)],
        "delta": delta,
        "comp_D": comp_D,
        "v0": v0,
        "fixed_S": fixed_S,
        "claimed_W_D_size": len(bres["W_D"]),
        "claimed_R": bres["R"],
        "claimed_Q_D_order": bres["Q_D_order"],
        "claimed_Q_D_primes": bres["Q_D_prime_divisors"],
        "density_audit": {
            "n0": n0,
            "m0": m0,
            "claimed_avg_degree": fmt_frac(avg_d),
            "claimed_thm11_regime": regime,
            "claimed_muller_exceeds_n_pow_n": mul_exceeds,
        },
    }
    if pillar3_cfg is not None:
        payload["pillar3_slice"] = pillar3_cfg
    return {
        "schema_version": SCHEMA_VERSION,
        "cert_id": cert_id,
        "mode": "bridge_rigidity_and_density_audit",
        "description": description,
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# Graph Family Constructors (Multi-Scale Hubs, Bridges, Twins, Pillar III)
# ---------------------------------------------------------------------------


def build_multiscale_hub_twin_graph() -> Tuple[int, List[Tuple[int, int]]]:
    """12-vertex connected multi-scale graph:
    - Hubs {0, 1, 2} form a triangle (K3) with additional attachments so d >= 5.
    - 3 twin K2-pendant copies attached identically to hub 0: {3,4}, {5,6}, {7,8}
      where 3,5,7 connect to hub 0 and {3,4},{5,6},{7,8} are internal edges -> 3! = 6 aut factor.
    - 1 bridge vertex {9} connecting hub 1 and hub 2 (deg=2, bridge component at Delta=4).
    - 2 open-neighborhood twin leaves {10, 11} both adjacent to {1, 2} -> 2! = 2 aut factor.
    Degrees: d(0)=5 ({1,2,3,5,7}), d(1)=5 ({0,2,9,10,11}), d(2)=5 ({0,1,9,10,11}),
             d(3)=d(5)=d(7)=2, d(4)=d(6)=d(8)=1, d(9)=2, d(10)=d(11)=2.
    """
    edges = [
        (0, 1), (0, 2), (1, 2),
        (0, 3), (3, 4),
        (0, 5), (5, 6),
        (0, 7), (7, 8),
        (1, 9), (2, 9),
        (1, 10), (2, 10),
        (1, 11), (2, 11),
    ]
    return 12, sorted(edges)


def build_two_tier_dyadic_graph() -> Tuple[int, List[Tuple[int, int]]]:
    """14-vertex 2-tier dyadic hierarchy (thresholds [3, 6, 12]):
    - Super-hub 0 (deg 9): connected to mid-hubs {1, 2}, plus 3 pendant leaves {11, 12, 13},
      plus 4-cycle bridge anchor {7, 8, 9, 10} -> let's construct explicitly:
    - Mid-hubs 1, 2 (deg 4 each): connected to 0, connected to each other? No, keep 1 and 2 distinct or twin:
      Let 0 connect to 1, 2, 3, 4, 5, 6, 7, 8 (deg 8).
      Let 1 connect to 0, 9, 10, 11 (deg 4).
      Let 2 connect to 0, 12, 13 (deg 3).
      Bridge edge (9, 12) joins {9, 12} across mid-hubs 1 and 2!
      Leaves: 10, 11 adjacent to 1 (twin leaves -> 2!); 3, 4, 5, 6, 7, 8 adjacent to 0:
      add edges (3, 4), (5, 6), (7, 8) -> three identical K2-triangles on hub 0! (3! = 6)
    Check degrees:
      d(0)=8 ({1,2,3,4,5,6,7,8}), d(1)=4 ({0,9,10,11}), d(2)=3 ({0,12,13}),
      d(3)=d(4)=d(5)=d(6)=d(7)=d(8)=2, d(9)=2 ({1,12}), d(12)=2 ({2,9}),
      d(10)=d(11)=1, d(13)=1.
    Connected, n=14, thresholds [3, 6, 9]:
      Delta_0 = 3: H_0={0,1,2}, U_0={1,2}, B_0={3..13};
        bridge comp {9,12} connects H_0 via {1,2}; non-bridge twin leaves {10},{11} on 1 (2!), {13} on 2;
      Delta_1 = 6: H_1={0}, U_1={0}, B_1={1..13};
        non-bridge {3,4},{5,6},{7,8} on 0 (3! = 6) + component {1,2,9,10,11,12,13}.
    """
    edges = [
        (0, 1), (0, 2),
        (0, 3), (0, 4), (3, 4),
        (0, 5), (0, 6), (5, 6),
        (0, 7), (0, 8), (7, 8),
        (1, 9), (1, 10), (1, 11),
        (2, 12), (2, 13),
        (9, 12),
    ]
    return 14, sorted(edges)


def build_three_tier_dyadic_graph() -> Tuple[int, List[Tuple[int, int]]]:
    """24-vertex 3-tier dyadic hierarchy (thresholds [3, 6, 12, 20]):
    - Top super-hub 0 (deg 13): connected to mid-hubs {1, 2} (deg 6 each), plus 4 copies of K2 attached to 0
      (8 vertices {3..10}: {3,4},{5,6},{7,8},{9,10} each forming a triangle with 0) -> 4! = 24,
      plus 3 pendant leaves {11, 12, 13} -> 3! = 6!
    - Mid-hub 1 (deg 6): connected to 0, to 2, to twin leaves {14, 15, 16} (deg 1 -> 3! = 6), and to bridge 17.
    - Mid-hub 2 (deg 6): connected to 0, to 1, to twin leaves {18, 19, 20} (deg 1 -> 3! = 6), and to bridge 21.
    - Bridge path 17 - 22 - 21 with extra leaf 23 on 22 (Wait: 22 has deg 3 if connected to 17, 21, 23 -> keep deg < 3:
      just bridge path 17 - 22 - 21 has deg(17)=2, deg(22)=2, deg(21)=2, and add twin leaf 23 to {1, 2}!).
    Check degrees:
      d(0) = 2 + 8 + 3 = 13
      d(1) = 1(0) + 1(2) + 3(14,15,16) + 1(17) + 1(23) = 7
      d(2) = 1(0) + 1(1) + 3(18,19,20) + 1(21) + 1(23) = 7
      d(3..10) = 2, d(11..13) = 1, d(14..16) = 1, d(18..20) = 1,
      d(17)=2, d(22)=2, d(21)=2, d(23)=2.
    Thresholds [3, 6, 12, 16]:
      H_0={0,1,2} (deg>=3), H_1={0,1,2} (deg>=6), H_2={0} (deg>=12).
      Better: add a tier-1 hub 23 with deg 4!
      Let's make vertex 23 have deg 4 by connecting 23 to 0, 1, 2, and a leaf? Wait n=24:
      Remove leaf 13 from 0 -> 13 connects to 23; connect 23 to 0, 1, 2, 13 (d(23)=4!).
      Then d(0) = 13 (1,2,23, 3..10, 11,12), d(1)=7 (0,2,23,14,15,16,17), d(2)=7 (0,1,23,18,19,20,21),
      d(23)=4 (0,1,2,13), d(13)=1.
      Now ALL THREE tiers are nonempty:
        U_0 = {23} (deg in [3,6)), U_1 = {1,2} (deg in [6,12)), U_2 = {0} (deg in [12,16))!
    """
    edges = [
        (0, 1), (0, 2), (1, 2),
        (0, 23), (1, 23), (2, 23), (13, 23),
        (0, 3), (0, 4), (3, 4),
        (0, 5), (0, 6), (5, 6),
        (0, 7), (0, 8), (7, 8),
        (0, 9), (0, 10), (9, 10),
        (0, 11), (0, 12),
        (1, 14), (1, 15), (1, 16),
        (2, 18), (2, 19), (2, 20),
        (1, 17), (17, 22), (22, 21), (2, 21),
    ]
    return 24, sorted(edges)


def build_bridge_rigidity_graph_star_d4() -> Tuple[int, List[Tuple[int, int]]]:
    """14-vertex graph testing Lemma 5.1(2) with Delta=4:
    - Hubs H_4 = {0, 1, 2, 3, 4, 5} form K_6 (each already has deg 5 >= 4).
    - Component D = {6, 7, 8, 9} in V_{<4}: tree rooted at v0=6 with children {7, 8, 9} (d(6)=3, d(7)=d(8)=d(9)=3):
      6 connects to 7, 8, 9 (deg_D(6)=3 < 4).
      7 connects to hubs {0, 1} (d_G(7) = 1 + 2 = 3 < 4).
      8 connects to hubs {2, 3} (d_G(8) = 1 + 2 = 3 < 4).
      9 connects to hubs {4, 5} (d_G(9) = 1 + 2 = 3 < 4).
    - Notice Q_D (with S = empty, R = 6) acts as wreath product S_2 wr S_3 of order 2^3 * 3! = 48!
      Prime divisors of 48 are {2, 3}, ALL strictly < Delta = 4, and 48 <= (16*4)^6 = 64^6!
    """
    edges: List[Tuple[int, int]] = []
    # K_6 on hubs 0..5
    for i in range(6):
        for j in range(i + 1, 6):
            edges.append((i, j))
    # Component D = {6, 7, 8, 9}
    edges.extend([(6, 7), (6, 8), (6, 9)])
    # Attachments to hubs
    edges.extend([(7, 0), (7, 1), (8, 2), (8, 3), (9, 4), (9, 5)])
    # Add 4 more isolated-hub connections? No, n=10 is already clean and exact!
    return 10, sorted(edges)


def build_bridge_rigidity_graph_d5_with_pillar3() -> Tuple[int, List[Tuple[int, int]], Dict[str, Any]]:
    """18-vertex graph combining Lemma 5.1(2) (Delta=5) and Pillar III (Prop. 8.4(2)):
    - Hubs H_5 = {0, 1, 2, 3, 4, 5} form K_6 (base deg 5 >= 5).
    - Bridge component D = {6, 7, 8, 9, 10}: star centered at v0=6 with 4 leaves {7,8,9,10}
      (d_G(6)=4 < 5):
        7 -> hub 0 (d=2), 8 -> hub 1 (d=2), 9 -> hub 2 (d=2), 10 -> hub 3 (d=2).
      Fixing S = {0}: W_D = {0,1,2,3}, W_free = {1,2,3}, R = 3, Q_D ~= S_3, |Q_D| = 6 (primes {2,3} < 5)!
    - Pillar III open-twin leaf classes of degree r_leaf = 2 attached to H_5:
        Class 0 (size 2): [11, 12] both adjacent to {0, 1}
        Class 1 (size 2): [13, 14] both adjacent to {2, 3}
        Class 2 (size 1): [15] adjacent to {4, 5}
        Class 3 (size 2): [16, 17] both adjacent to {0, 4}
      Select indices [0, 1] (2/4 = 1/2 >= 3/8): W_0 = {11, 12, 13, 14}, |W_0| = 4.
      Every hub in H_5 has d_G >= 6 and loses at most 2 edges in K_0 = G - W_0, so 2*d_{K0} >= d_G!
      Total degree excess across H_5 = 2 + 2 + 2 + 2 = 8 == r_leaf * |W_0| = 2 * 4!
    """
    edges: List[Tuple[int, int]] = []
    for i in range(6):
        for j in range(i + 1, 6):
            edges.append((i, j))
    # Component D = {6, 7, 8, 9, 10}
    edges.extend([(6, 7), (6, 8), (6, 9), (6, 10)])
    edges.extend([(7, 0), (8, 1), (9, 2), (10, 3)])
    # Pillar III leaf classes (all degree 2, all neighbors in H_5)
    edges.extend([
        (11, 0), (11, 1), (12, 0), (12, 1),
        (13, 2), (13, 3), (14, 2), (14, 3),
        (15, 4), (15, 5),
        (16, 0), (16, 4), (17, 0), (17, 4),
    ])
    p3 = {
        "delta_prev": 5,
        "r_leaf": 2,
        "twin_classes": [[11, 12], [13, 14], [15], [16, 17]],
        "selected_class_indices": [0, 1],
        "claimed_total_excess": 8,
    }
    return 18, sorted(edges), p3


def build_bridge_rigidity_graph_d6_with_pillar3() -> Tuple[int, List[Tuple[int, int]], Dict[str, Any]]:
    """22-vertex graph combining Lemma 5.1(2) (Delta=6) and Pillar III (r_leaf=3):
    - Hubs H_6 = {0..7} form K_8 (base deg 7 >= 6).
    - Bridge component D = {8, 9, 10, 11, 12, 13}: center v0=8 connected to 5 leaves {9,10,11,12,13} (d(8)=5 < 6),
      leaf 9->hub 0, 10->hub 1, 11->hub 2, 12->hub 3, 13->hub 4.
      With S = {0}, W_free = {1,2,3,4}, R = 4, |Q_D| = 4! = 24 (primes {2,3} < 6)!
    - Pillar III open-twin leaf classes of degree r_leaf = 3 attached to H_6:
        Class 0 (size 2): [14, 15] -> {0, 1, 2}
        Class 1 (size 2): [16, 17] -> {3, 4, 5}
        Class 2 (size 2): [18, 19] -> {5, 6, 7}
        Class 3 (size 1): [20] -> {0, 6, 7}
        Class 4 (size 1): [21] -> {1, 2, 3}
      Select indices [0, 1] (2/5 = 0.4 >= 3/8 = 0.375): W_0 = {14, 15, 16, 17}, |W_0| = 4.
      Total degree excess = 3 * 4 = 12, every hub loses <= 2 out of >= 8 edges -> 2 d_K0 >= d_G!
    """
    edges: List[Tuple[int, int]] = []
    for i in range(8):
        for j in range(i + 1, 8):
            edges.append((i, j))
    edges.extend([(8, 9), (8, 10), (8, 11), (8, 12), (8, 13)])
    edges.extend([(9, 0), (10, 1), (11, 2), (12, 3), (13, 4)])
    edges.extend([
        (14, 0), (14, 1), (14, 2), (15, 0), (15, 1), (15, 2),
        (16, 3), (16, 4), (16, 5), (17, 3), (17, 4), (17, 5),
        (18, 5), (18, 6), (18, 7), (19, 5), (19, 6), (19, 7),
        (20, 0), (20, 6), (20, 7),
        (21, 1), (21, 2), (21, 3),
    ])
    p3 = {
        "delta_prev": 6,
        "r_leaf": 3,
        "twin_classes": [[14, 15], [16, 17], [18, 19], [20], [21]],
        "selected_class_indices": [0, 1],
        "claimed_total_excess": 12,
    }
    return 22, sorted(edges), p3


def build_all_16_certificates() -> List[Dict[str, Any]]:
    certs: List[Dict[str, Any]] = []

    # ---- MODE 1: 4 Canonical Certificates (Harary sharp boundary + Kelly + Greenwell) ----
    # 1. Harary sharp counterexample at m = 2: 2K2 vs K_{1,2} cup K_1 (n = 4)
    certs.append(
        make_mode1_cert(
            "m1_harary_m2_counterexample",
            "Classical m=2 non-edge-reconstructible pair: 2K2 vs K_{1,2} U K1 (identical edge-deck {K2 U 2K1}^2)",
            n=4,
            edges_G=[(0, 1), (2, 3)],
            edges_H=[(0, 1), (0, 2)],
            verdict="NON_ISOMORPHIC_COUNTEREXAMPLE",
            kelly_patterns=["K2"],
        )
    )
    # 2. Harary sharp counterexample at m = 3: K3 U K1 vs K_{1,3} (n = 4)
    certs.append(
        make_mode1_cert(
            "m1_harary_m3_counterexample",
            "Classical m=3 non-edge-reconstructible pair: K3 U K1 vs claw K_{1,3} (identical edge-deck {K_{1,2} U K1}^3)",
            n=4,
            edges_G=[(0, 1), (1, 2), (0, 2)],
            edges_H=[(0, 1), (0, 2), (0, 3)],
            verdict="NON_ISOMORPHIC_COUNTEREXAMPLE",
            kelly_patterns=["K2", "P3"],
        )
    )
    # 3. Isomorphic dense graph on n=6, m=11 (passes Müller 2^10=1024 > 720=6!)
    # K6 minus 4 edges: remove (0,5),(1,4),(2,3),(0,4) -> m=11
    k6_minus4 = [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (1, 5),
        (2, 4), (2, 5), (3, 4), (3, 5), (4, 5),
    ]
    phi6 = [2, 5, 0, 4, 1, 3]
    k6_perm = sorted((min(phi6[u], phi6[v]), max(phi6[u], phi6[v])) for u, v in k6_minus4)
    certs.append(
        make_mode1_cert(
            "m1_muller_dense_n6_iso_kelly",
            "Isomorphic n=6, m=11 pair exceeding Muller threshold (2^10=1024 > 6!=720) with full Kelly & Greenwell audit",
            n=6,
            edges_G=k6_minus4,
            edges_H=k6_perm,
            verdict="ISOMORPHIC_DECK_MATCH",
            kelly_patterns=["K2", "P3", "K3", "C4", "K1_3"],
            isomorphism_phi=phi6,
        )
    )
    # 4. Non-isomorphic 5-vertex graphs with m=5 edges (past Harary m>=4: decks MUST disagree!)
    # House graph vs Bull graph (both n=5, m=5, isolate-free, connected)
    house_edges = [(0, 1), (1, 2), (2, 3), (0, 3), (2, 4)]  # P3+C4? Wait: C4=(0,1,2,3), roof=(2,4),(3,4) has 6 edges!
    # Use 5-vertex 5-edge connected graphs: C5 vs Paw+pendant (triangle + tail of length 2)
    c5_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
    pan_tail_edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4)]
    certs.append(
        make_mode1_cert(
            "m1_harary_m5_deck_distinguished",
            "Distinct connected isolate-free graphs at n=5, m=5 (C5 vs lollipop): edge-deck & vertex-deck separate them",
            n=5,
            edges_G=c5_edges,
            edges_H=pan_tail_edges,
            verdict="DECK_MISMATCH",
            kelly_patterns=["K2", "P3", "K3", "P4"],
        )
    )

    # ---- MODE 2: 4 Canonical Certificates (Single Kruskal MST Multi-Scale Rigidity) ----
    # 5. Multi-scale hub + twin-K2 + bridge + open-twin leaf graph (n=12, exact Aut checked!)
    n12, e12 = build_multiscale_hub_twin_graph()
    certs.append(
        make_mode2_cert(
            "m2_kruskal_n12_hub_twins_exact_aut",
            "12-vertex connected hub/bridge/twin graph verifying Lemma 3.1(1)-(2) with exact |Aut(G)| divisibility",
            n=n12,
            edges=e12,
            thresholds=[3, 6],
            phi_delta_scale=3,
            include_exact_aut=True,
        )
    )
    # 6. Two-tier dyadic hierarchy (n=14, thresholds [3, 6, 9], exact Aut checked!)
    n14, e14 = build_two_tier_dyadic_graph()
    certs.append(
        make_mode2_cert(
            "m2_kruskal_n14_two_tier_dyadic",
            "14-vertex 2-tier dyadic Kruskal hierarchy (Delta in {3,6,9}) with multi-scale subgroup factorization",
            n=n14,
            edges=e14,
            thresholds=[3, 6, 9],
            phi_delta_scale=3,
            include_exact_aut=True,
        )
    )
    # 7. Three-tier dyadic hierarchy (n=24, thresholds [3, 6, 12, 16])
    n24, e24 = build_three_tier_dyadic_graph()
    certs.append(
        make_mode2_cert(
            "m2_kruskal_n24_three_tier_dyadic",
            "24-vertex 3-tier dyadic Kruskal hierarchy (Delta in {3,6,12,16}) with bridge path and 4!*3!*3!*2! twin product",
            n=n24,
            edges=e24,
            thresholds=[3, 6, 12, 16],
            phi_delta_scale=6,
            include_exact_aut=False,
        )
    )
    # 8. Dense K6-bridge-K6 barbell with twin stars (n=20, thresholds [4, 8, 12])
    edges20: List[Tuple[int, int]] = []
    # Two K5 cliques {0..4} and {5..9} joined by two degree-2 bridge paths (0-10-5) and (1-11-6)
    for i in range(5):
        for j in range(i + 1, 5):
            edges20.append((i, j))
    for i in range(5, 10):
        for j in range(i + 1, 10):
            edges20.append((i, j))
    edges20.extend([(0, 10), (10, 5), (1, 11), (11, 6)])
    # Attach 4 twin leaves {12,13,14,15} to hub 0 (giving d(0)=9) and 4 twin leaves {16,17,18,19} to hub 5 (d(5)=9)
    for w in [12, 13, 14, 15]:
        edges20.append((0, w))
    for w in [16, 17, 18, 19]:
        edges20.append((5, w))
    certs.append(
        make_mode2_cert(
            "m2_kruskal_n20_dual_cliques_bridges",
            "20-vertex dual-clique multi-bridge hierarchy (Delta in {3,6,10}) verifying |B_Delta|<=|H|-1 & Phi_Delta",
            n=20,
            edges=sorted(edges20),
            thresholds=[3, 6, 10],
            phi_delta_scale=3,
            include_exact_aut=False,
        )
    )

    # ---- MODE 3: 4 Canonical Certificates (Shift-Invariant Polynomials, Charlier, Lemma 4.3) ----
    # 9. Sharp t=1 Vandermonde-style multilinear shift-invariant poly on M=4 vars, N0=2:
    # P(y0,y1,y2,y3) = (y0 - y1)(y2 - y3) -> target v=(1,0,1,0), t_max=1, ||P||_1=4, |c_v|=1, ratio=4 >= 4/3!
    p_m3_1 = poly_prod([diff_linear(4, 0, 1), diff_linear(4, 2, 3)], 4)
    certs.append(
        make_mode3_cert(
            "m3_shift_t1_multilinear_n2",
            "Sharp t=1 shift-invariant multilinear polynomial (y0-y1)(y2-y3) with exact Lemma 4.1(1) Charlier bound",
            M=4,
            poly=p_m3_1,
            target_v=(1, 0, 1, 0),
            t_max=1,
            charlier_d=1,
            charlier_r=6,
            charlier_kmax=12,
        )
    )
    # 10. Sharp t=1 higher degree N0=4 on M=8 vars: (y0-y1)(y2-y3)(y4-y5)(y6-y7), ratio = 16 >= (4/3)^2 = 16/9
    p_m3_2 = poly_prod(
        [diff_linear(8, 0, 1), diff_linear(8, 2, 3), diff_linear(8, 4, 5), diff_linear(8, 6, 7)],
        8,
    )
    certs.append(
        make_mode3_cert(
            "m3_shift_t1_multilinear_n4",
            "Degree-4 sharp t=1 shift-invariant polynomial in 8 variables (ratio 16) + Charlier d=3, r=10",
            M=8,
            poly=p_m3_2,
            target_v=(1, 0, 1, 0, 1, 0, 1, 0),
            t_max=1,
            charlier_d=3,
            charlier_r=10,
            charlier_kmax=16,
        )
    )
    # 11. Uniform multiplicity-partition shift-invariant polynomial with Lemma 4.3 band isolation!
    # Need EVERY u in supp(P) to be a permutation of v!
    # Consider antisymmetric Vandermonde polynomial V(y0, y1, y2) = (y0-y1)(y0-y2)(y1-y2):
    # EVERY monomial in V(y0,y1,y2) is a permutation of (2, 1, 0)!! AND V is shift-invariant (D_I V == 0)!
    # Even better: 4-variable Vandermonde V(y0,y1,y2,y3) has EVERY monomial a permutation of (3,2,1,0)!
    # For V(y0,y1,y2,y3): degree N0=6, 24 terms, coeffs +-1, ||V||_1=24, |c_v|=1, ratio=24!
    # Band isolation [t0, t1] = [1, 2]:
    #   I_lo = {3} (v_3=0 < 1), I_mid = {1, 2} (v_1=2, v_2=1 in [1,2]), I_hi = {0} (v_0=3 > 2)!
    #   C_const = C(3,1)*C(2,1)*C(1,1) = 6, F_mid(y1, y2) has degree (2-1)+(1-1) = 1, F_mid = y1 - y2, D_{I_mid}(F_mid) = 0!
    vdm4 = poly_prod(
        [
            diff_linear(4, 0, 1), diff_linear(4, 0, 2), diff_linear(4, 0, 3),
            diff_linear(4, 1, 2), diff_linear(4, 1, 3), diff_linear(4, 2, 3),
        ],
        4,
    )
    certs.append(
        make_mode3_cert(
            "m3_vandermonde4_lemma43_band",
            "4-variable Vandermonde shift-invariant polynomial certifying Lemma 4.3 band isolation [1,2] & Thm 4.2",
            M=4,
            poly=vdm4,
            target_v=(3, 2, 1, 0),
            t_max=3,
            charlier_d=4,
            charlier_r=12,
            charlier_kmax=18,
            band_bounds=(1, 2),
        )
    )
    # 12. 5-variable Vandermonde V(y0..y4): degree N0=10, 120 terms, EVERY monomial permutation of (4,3,2,1,0)!
    # Band isolation [t0, t1] = [1, 3]:
    #   I_lo = {4} (v=0), I_mid = {1, 2, 3} (v=3,2,1), I_hi = {0} (v=4).
    #   F_mid is a degree-3 shift-invariant Vandermonde in 3 variables (||F_mid||_1 = 6 <= 120)!
    vdm5 = poly_prod(
        [diff_linear(5, i, j) for i in range(5) for j in range(i + 1, 5)],
        5,
    )
    certs.append(
        make_mode3_cert(
            "m3_vandermonde5_lemma43_band",
            "5-variable Vandermonde (120 terms, deg 10) certifying Lemma 4.3 band [1,3] reduction to deg-3 Vandermonde",
            M=5,
            poly=vdm5,
            target_v=(4, 3, 2, 1, 0),
            t_max=4,
            charlier_d=5,
            charlier_r=16,
            charlier_kmax=22,
            band_bounds=(1, 3),
        )
    )

    # ---- MODE 4: 4 Canonical Certificates (Lemma 5.1 Bridge Rigidity + Pillar III + Thm 1.1) ----
    # 13. Bridge rigidity star D (Delta=4, |Q_D|=48, primes {2,3}<4) + Vacuous regime n0=2000
    n_b1, e_b1 = build_bridge_rigidity_graph_star_d4()
    certs.append(
        make_mode4_cert(
            "m4_bridge_d4_wreath48_vacuous2000",
            "Lemma 5.1(2) wreath-product bridge group |Q_D|=48 (Delta=4, R=6) + Thm 1.1 vacuous boundary n0=2000",
            n=n_b1,
            edges=e_b1,
            delta=4,
            comp_D=[6, 7, 8, 9],
            v0=6,
            fixed_S=[],
            n0=2000,
            m0=1_999_000,  # C(2000,2) = 1,999,000 < 2,000,000 = 1000*2000
        )
    )
    # 14. Bridge rigidity (Delta=5, |Q_D|=6) + Pillar III r_leaf=2 (exact excess 8) + Active n0=2001, m0=2,001,000
    n_b2, e_b2, p3_2 = build_bridge_rigidity_graph_d5_with_pillar3()
    certs.append(
        make_mode4_cert(
            "m4_bridge_d5_pillar3_r2_active2001",
            "Lemma 5.1(2) (Delta=5, |Q_D|=6) + Prop. 8.4 Pillar III (r=2, excess 8) + Thm 1.1 onset n0=2001",
            n=n_b2,
            edges=e_b2,
            delta=5,
            comp_D=[6, 7, 8, 9, 10],
            v0=6,
            fixed_S=[0],
            n0=2001,
            m0=2_001_000,  # K_2001 has exactly 2001*1000 edges!
            pillar3_cfg=p3_2,
        )
    )
    # 15. Bridge rigidity (Delta=6, |Q_D|=24) + Pillar III r_leaf=3 (exact excess 12) + Linear-density n0=10^6, m0=10^9
    n_b3, e_b3, p3_3 = build_bridge_rigidity_graph_d6_with_pillar3()
    certs.append(
        make_mode4_cert(
            "m4_bridge_d6_pillar3_r3_active1e6",
            "Lemma 5.1(2) (Delta=6, |Q_D|=24) + Prop. 8.4 Pillar III (r=3, excess 12) + active linear density n0=10^6",
            n=n_b3,
            edges=e_b3,
            delta=6,
            comp_D=[8, 9, 10, 11, 12, 13],
            v0=8,
            fixed_S=[0],
            n0=1_000_000,
            m0=1_000_000_000,
            pillar3_cfg=p3_3,
        )
    )
    # 16. Bridge rigidity (Delta=6, S={0,1}, R=3, |Q_D|=6) + Super-exponential n0=2^1005 beating Muller!
    # At n0 = 2^1005 (~2.1e302), m0 = 1000 * n0 STILL satisfies Theorem 1.1 (ACTIVE_LINEAR_DENSITY)
    # while m0 - 1 < 1005 * n0 so Muller's n0 log2 n0 FAILS!!
    n0_huge = 1 << 1005
    m0_huge = 1000 * n0_huge
    certs.append(
        make_mode4_cert(
            "m4_bridge_d6_beyond_muller_2pow1005",
            "Active Theorem 1.1 regime at n0=2^1005 (m0=1000 n0) strictly below Muller's n0 log2 n0 threshold!",
            n=n_b3,
            edges=e_b3,
            delta=6,
            comp_D=[8, 9, 10, 11, 12, 13],
            v0=8,
            fixed_S=[0, 1],
            n0=n0_huge,
            m0=m0_huge,
        )
    )

    return certs


def main() -> None:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    # Clean stale cert files if any
    for old in CERT_DIR.glob("*.cert.json"):
        old.unlink()

    certs = build_all_16_certificates()
    assert len(certs) == 16, f"Expected 16 certs, got {len(certs)}"

    # Exp 1: Write, verify, and benchmark all 16 canonical certificates (7-run mean after warm-up)
    atlas_rows: List[Dict[str, Any]] = []
    for cert in certs:
        out_path = CERT_DIR / f"{cert['cert_id']}.cert.json"
        raw = json.dumps(cert, indent=2) + "\n"
        out_path.write_text(raw, encoding="utf-8")

        # Warm-up verification
        res = verify_certificate_dict(cert)
        assert res["status"] == "VERIFIED"

        times_ms: List[float] = []
        for _ in range(7):
            t0 = time.perf_counter()
            verify_certificate_dict(cert)
            times_ms.append((time.perf_counter() - t0) * 1000.0)
        mean_ms = round(sum(times_ms) / len(times_ms), 3)
        atlas_rows.append(
            {
                "cert_id": cert["cert_id"],
                "mode": cert["mode"],
                "mean_ms": mean_ms,
                "min_ms": round(min(times_ms), 3),
                "max_ms": round(max(times_ms), 3),
                "summary": res["summary"],
            }
        )
        print(f"[VERIFIED] {cert['cert_id']:40s} ({cert['mode']:34s}) {mean_ms:7.3f} ms")

    (RES_DIR / "exp1_certificate_atlas.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "count": len(atlas_rows), "certificates": atlas_rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    # Exp 2: Density threshold comparison (Lovász 1972 vs Müller 1977 vs Cogentic 2026 vs Remark 1.3 barrier)
    density_rows: List[Dict[str, Any]] = []
    for exp_k in [4, 8, 12, 20, 50, 100, 250, 500, 1000, 1001, 1005, 1500, 2000, 4000]:
        # Average degree thresholds (2m/n) as functions of log2(n) = exp_k
        d_muller = 2.0 * exp_k  # m > n log2 n => avg degree > 2 log2 n
        d_cogentic = 2000.0  # m >= 1000 n => avg degree >= 2000
        d_barrier_reg = 8.0
        d_barrier_ham = 7.32
        density_rows.append(
            {
                "log2_n": exp_k,
                "muller_avg_deg_2log2n": d_muller,
                "cogentic_avg_deg": d_cogentic,
                "cogentic_beats_muller": d_cogentic < d_muller,
                "ratio_muller_over_cogentic": round(d_muller / d_cogentic, 4),
                "barrier_d8": d_barrier_reg,
                "barrier_d_star": d_barrier_ham,
            }
        )
    # Exact Remark 1.3 gap table for small integer average degrees d = 2..14
    remark13_rows: List[Dict[str, Any]] = []
    for d in range(2, 15):
        parity_bits_per_v = d / 2.0 - 1.0
        deg_ub_bits_per_v = log2(d)
        ham_ub_bits_per_v = log2(d - 1) if d >= 2 else 0.0
        remark13_rows.append(
            {
                "avg_degree_d": d,
                "parity_bits_per_vertex": round(parity_bits_per_v, 6),
                "log2_d_bits_per_vertex": round(deg_ub_bits_per_v, 6),
                "log2_d_minus_1_bits_per_vertex": round(ham_ub_bits_per_v, 6),
                "parity_exceeds_log2_d": parity_bits_per_v > deg_ub_bits_per_v,
                "parity_exceeds_log2_d_minus_1": parity_bits_per_v > ham_ub_bits_per_v,
            }
        )
    (RES_DIR / "exp2_density_and_barrier.json").write_text(
        json.dumps(
            {
                "crossover_log2_n": 1000,
                "vacuous_upper_n": 2000,
                "first_active_n": 2001,
                "density_sweep": density_rows,
                "remark13_small_d": remark13_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # Exp 3: Exact Charlier I_{d,k}(r; 1/(2r)) profiles vs Lemma 4.1(1) bound 1 - 11d/(24r)
    charlier_profiles: List[Dict[str, Any]] = []
    for (d_val, r_val) in [(1, 6), (2, 8), (3, 10), (4, 12), (5, 16), (7, 20)]:
        lam = Fraction(1, 2 * r_val)
        ub = Fraction(1, 1) - Fraction(11 * d_val, 24 * r_val)
        vals: List[Dict[str, Any]] = []
        max_abs = Fraction(0, 1)
        max_k = 0
        for k_val in range(0, 2 * d_val + 8):
            val = charlier_scaled_exact(d_val, k_val, r_val, lam)
            aval = abs(val)
            if aval > max_abs:
                max_abs = aval
                max_k = k_val
            vals.append(
                {
                    "k": k_val,
                    "exact_I": fmt_frac(val),
                    "abs_I_float": round(float(aval), 8),
                }
            )
        assert max_abs <= ub
        charlier_profiles.append(
            {
                "d": d_val,
                "r": r_val,
                "lambda_exact": fmt_frac(lam),
                "upper_bound_exact": fmt_frac(ub),
                "upper_bound_float": round(float(ub), 8),
                "max_abs_I_exact": fmt_frac(max_abs),
                "max_abs_I_float": round(float(max_abs), 8),
                "argmax_k": max_k,
                "margin_float": round(float(ub - max_abs), 8),
                "curve": vals,
            }
        )
    (RES_DIR / "exp3_charlier_profiles.json").write_text(
        json.dumps({"profiles": charlier_profiles}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Exp 4: Exact Lemma 5.1 Prime-Sum Bound sum_{p<=Delta} ln(p)/(p-1) vs ln(16*Delta)
    # and Kruskal bridge-bound saturation ratios across our verified multi-scale graphs
    def is_prime(x: int) -> bool:
        if x < 2:
            return False
        d = 2
        while d * d <= x:
            if x % d == 0:
                return False
            d += 1
        return True

    from math import log as ln_f

    prime_sum_rows: List[Dict[str, Any]] = []
    for delta_val in [2, 3, 4, 5, 7, 10, 16, 32, 64, 128, 256, 512, 1024, 2000]:
        psum = sum(ln_f(p) / (p - 1) for p in range(2, delta_val + 1) if is_prime(p))
        ub_ln = ln_f(16 * delta_val)
        assert psum <= ub_ln
        prime_sum_rows.append(
            {
                "delta": delta_val,
                "prime_sum": round(psum, 6),
                "ln_16_delta": round(ub_ln, 6),
                "slack": round(ub_ln - psum, 6),
            }
        )
    (RES_DIR / "exp4_prime_sum_and_rigidity.json").write_text(
        json.dumps({"prime_sum_sweep": prime_sum_rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nGenerated 16 certificates and 4 result JSONs successfully.")


if __name__ == "__main__":
    main()
