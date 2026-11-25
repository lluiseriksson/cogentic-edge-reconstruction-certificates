"""Zero-float 4-mode certificate verifier for Paper 60 (EDGERECON-COGENTIC-CERT-1.0).

Verifies exact combinatorics, multi-scale Kruskal rigidity, shift-invariant Charlier
polynomials & multiplicity-band isolation, and bounded-degree bridge automorphism
rigidity / Pillar III degree-excess conservation for the Cogentic (2026) unconditional
linear-density edge reconstruction theorem (m >= 1000 n_0).

Strictly zero floating-point arithmetic (`float`, `math.log`, `numpy` forbidden).
"""

from __future__ import annotations

import json
from fractions import Fraction
from math import comb, factorial
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from src.edgerecon_exact import (
    STANDARD_PATTERNS,
    analyze_kruskal_scale,
    are_isomorphic,
    automorphism_order,
    build_adj,
    charlier_scaled_exact,
    compute_bridge_restriction_group,
    connected_components,
    count_induced_subgraphs,
    count_subgraphs,
    decks_equal,
    degree_sequence,
    degrees,
    edge_deck,
    fmt_frac,
    is_connected,
    kruskal_degree_ordered_mst,
    lemma43_band_isolation,
    parse_exact_fraction,
    parse_exact_int,
    parse_sparse_poly,
    phi_delta_functional,
    poly_directional_derivative,
    poly_l1_norm,
    validate_simple_graph,
    verify_isomorphism_map,
    vertex_deck,
)


SCHEMA_VERSION = "EDGERECON-COGENTIC-CERT-1.0"
MAX_JSON_DEPTH = 16
MAX_JSON_BYTES = 2_000_000

VALID_MODES = {
    "edge_deck_hypomorphism_and_kelly",
    "kruskal_multiscale_rigidity_cert",
    "shift_invariant_charlier_band_cert",
    "bridge_rigidity_and_density_audit",
}


class CertificateError(ValueError):
    """Raised when a certificate fails structural, parsing, or mathematical verification."""


# ---------------------------------------------------------------------------
# Strict JSON Loader (Duplicate-Key Rejection, Zero-Float, Depth Limit)
# ---------------------------------------------------------------------------


def _no_duplicate_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    obj: Dict[str, Any] = {}
    for k, v in pairs:
        if k in obj:
            raise CertificateError(f"Duplicate JSON key detected: {k!r}")
        obj[k] = v
    return obj


def _reject_float_constant(val: str) -> Any:
    raise CertificateError(f"Floating-point constant rejected in strict zero-float certificate: {val!r}")


def _check_depth_and_types(obj: Any, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise CertificateError(f"JSON nesting depth exceeded maximum {MAX_JSON_DEPTH}")
    if isinstance(obj, float):
        raise CertificateError(f"Floating-point value {obj!r} rejected in strict zero-float certificate")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise CertificateError(f"Non-string JSON object key: {k!r}")
            _check_depth_and_types(v, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _check_depth_and_types(item, depth + 1)


def load_strict_json(raw: str) -> Dict[str, Any]:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_JSON_BYTES:
        raise CertificateError("Certificate input must be a UTF-8 string within size limit")
    try:
        data = json.loads(
            raw,
            object_pairs_hook=_no_duplicate_pairs,
            parse_float=_reject_float_constant,
            parse_constant=_reject_float_constant,
        )
    except CertificateError:
        raise
    except Exception as exc:
        raise CertificateError(f"Invalid JSON syntax: {exc}") from exc
    if not isinstance(data, dict):
        raise CertificateError("Top-level certificate must be a JSON object")
    _check_depth_and_types(data)
    return data


def _require_keys(obj: Dict[str, Any], required: Set[str], optional: Set[str], ctx: str) -> None:
    keys = set(obj.keys())
    missing = required - keys
    extra = keys - (required | optional)
    if missing:
        raise CertificateError(f"{ctx}: missing required keys {sorted(missing)}")
    if extra:
        raise CertificateError(f"{ctx}: unexpected extra keys {sorted(extra)}")


def _require_bool(val: Any, field: str) -> bool:
    if not isinstance(val, bool):
        raise CertificateError(f"{field}: expected boolean, got {type(val).__name__}")
    return val


def _require_str(val: Any, field: str) -> str:
    if not isinstance(val, str) or not val.strip():
        raise CertificateError(f"{field}: expected non-empty string")
    return val


# ---------------------------------------------------------------------------
# Mode 1: edge_deck_hypomorphism_and_kelly (Sec. 1.1 & Sec. 2.1-2.2)
# ---------------------------------------------------------------------------


def _verify_mode1(payload: Dict[str, Any]) -> Dict[str, Any]:
    _require_keys(
        payload,
        required={
            "n",
            "edges_G",
            "edges_H",
            "verdict",
            "claimed_edge_decks_equal",
            "claimed_vertex_decks_equal",
            "claimed_aut_G",
            "claimed_aut_H",
            "kelly_patterns",
            "claimed_kelly_edge_counts_G",
            "claimed_kelly_vertex_counts_G",
            "claimed_muller_holds",
        },
        optional={"isomorphism_phi", "note"},
        ctx="mode1.payload",
    )

    n = parse_exact_int(payload["n"], "n")
    if n < 4 or n > 10:
        raise CertificateError(f"Mode 1 requires 4 <= n <= 10 for exhaustive exact deck audit, got n={n}")

    _, edges_G = validate_simple_graph(n, payload["edges_G"], "edges_G")
    _, edges_H = validate_simple_graph(n, payload["edges_H"], "edges_H")
    adj_G = build_adj(n, edges_G)
    adj_H = build_adj(n, edges_H)
    m_G = len(edges_G)
    m_H = len(edges_H)

    verdict = _require_str(payload["verdict"], "verdict")
    if verdict not in {"NON_ISOMORPHIC_COUNTEREXAMPLE", "ISOMORPHIC_DECK_MATCH", "DECK_MISMATCH"}:
        raise CertificateError(f"Invalid Mode 1 verdict: {verdict!r}")

    claimed_ed = _require_bool(payload["claimed_edge_decks_equal"], "claimed_edge_decks_equal")
    claimed_vd = _require_bool(payload["claimed_vertex_decks_equal"], "claimed_vertex_decks_equal")
    claimed_aut_G = parse_exact_int(payload["claimed_aut_G"], "claimed_aut_G")
    claimed_aut_H = parse_exact_int(payload["claimed_aut_H"], "claimed_aut_H")
    claimed_muller = _require_bool(payload["claimed_muller_holds"], "claimed_muller_holds")

    # Exact isomorphism & automorphism orders
    iso_exists = are_isomorphic(n, adj_G, adj_H)
    aut_G = automorphism_order(n, adj_G)
    aut_H = automorphism_order(n, adj_H)
    if aut_G != claimed_aut_G:
        raise CertificateError(f"claimed_aut_G mismatch: claimed {claimed_aut_G}, actual {aut_G}")
    if aut_H != claimed_aut_H:
        raise CertificateError(f"claimed_aut_H mismatch: claimed {claimed_aut_H}, actual {aut_H}")

    # Exact edge-deck and vertex-deck multiset equality
    ed_G = edge_deck(n, edges_G)
    ed_H = edge_deck(n, edges_H)
    vd_G = vertex_deck(n, edges_G)
    vd_H = vertex_deck(n, edges_H)
    actual_ed = decks_equal(ed_G, ed_H)
    actual_vd = decks_equal(vd_G, vd_H)
    if actual_ed != claimed_ed:
        raise CertificateError(f"claimed_edge_decks_equal mismatch: claimed {claimed_ed}, actual {actual_ed}")
    if actual_vd != claimed_vd:
        raise CertificateError(f"claimed_vertex_decks_equal mismatch: claimed {claimed_vd}, actual {actual_vd}")

    # Exact verdict semantics
    if verdict == "ISOMORPHIC_DECK_MATCH":
        if "isomorphism_phi" not in payload:
            raise CertificateError("ISOMORPHIC_DECK_MATCH requires explicit witness isomorphism_phi")
        phi_raw = payload["isomorphism_phi"]
        if not isinstance(phi_raw, list) or len(phi_raw) != n:
            raise CertificateError("isomorphism_phi must be a length-n permutation list")
        phi = [parse_exact_int(x, f"isomorphism_phi[{i}]") for i, x in enumerate(phi_raw)]
        if not verify_isomorphism_map(n, adj_G, adj_H, phi):
            raise CertificateError("isomorphism_phi fails exact edge-preservation G <-> H")
        if not (actual_ed and actual_vd):
            raise CertificateError("Isomorphic graphs must have identical edge-decks and vertex-decks")
    elif verdict == "NON_ISOMORPHIC_COUNTEREXAMPLE":
        if iso_exists:
            raise CertificateError("NON_ISOMORPHIC_COUNTEREXAMPLE rejected: G and H are actually isomorphic")
        if not actual_ed:
            raise CertificateError("NON_ISOMORPHIC_COUNTEREXAMPLE rejected: edge-decks D_E(G) != D_E(H)")
        if m_G > 3:
            raise CertificateError(f"NON_ISOMORPHIC_COUNTEREXAMPLE unexpected at m={m_G} > 3 (Harary boundary)")
    else:  # DECK_MISMATCH
        if iso_exists:
            raise CertificateError("DECK_MISMATCH rejected: G and H are actually isomorphic")
        if actual_ed:
            raise CertificateError("DECK_MISMATCH rejected: edge-decks actually match")

    # Greenwell reduction sanity check (for isolate-free m >= 4, D_E(G)=D_E(H) => D(G)=D(H))
    iso_v_G = sum(1 for nbrs in adj_G if len(nbrs) == 0)
    iso_v_H = sum(1 for nbrs in adj_H if len(nbrs) == 0)
    if m_G >= 4 and iso_v_G == 0 and iso_v_H == 0 and actual_ed and not actual_vd:
        raise CertificateError("Greenwell edge-to-vertex deck implication violated")

    # Exact Kelly's Lemma Z-identities on G (edge-deleted & vertex-deleted)
    patterns = payload["kelly_patterns"]
    if not isinstance(patterns, list) or not patterns:
        raise CertificateError("kelly_patterns must be a non-empty list")
    if len(set(patterns)) != len(patterns):
        raise CertificateError("Duplicate entries in kelly_patterns")
    claimed_ke = payload["claimed_kelly_edge_counts_G"]
    claimed_kv = payload["claimed_kelly_vertex_counts_G"]
    if not isinstance(claimed_ke, dict) or not isinstance(claimed_kv, dict):
        raise CertificateError("claimed_kelly_edge_counts_G and claimed_kelly_vertex_counts_G must be objects")
    if set(claimed_ke.keys()) != set(patterns) or set(claimed_kv.keys()) != set(patterns):
        raise CertificateError("Kelly count maps must have exact keys matching kelly_patterns")

    kelly_summary: Dict[str, Any] = {}
    for pat in patterns:
        if pat not in STANDARD_PATTERNS:
            raise CertificateError(f"Unknown Kelly pattern: {pat!r}")
        pn, pedges = STANDARD_PATTERNS[pat]
        pm = len(pedges)

        # Edge-Kelly: (m_G - pm) * s(F, G) == sum_{e in E(G)} s(F, G - e) when pm < m_G
        s_val = count_subgraphs(n, edges_G, pat)
        cl_s = parse_exact_int(claimed_ke[pat], f"claimed_kelly_edge_counts_G[{pat}]")
        if s_val != cl_s:
            raise CertificateError(f"Kelly edge count mismatch for {pat}: claimed {cl_s}, actual {s_val}")
        if pm < m_G:
            card_sum_e = sum(
                count_subgraphs(n, [e for j, e in enumerate(edges_G) if j != i], pat)
                for i in range(m_G)
            )
            if (m_G - pm) * s_val != card_sum_e:
                raise CertificateError(
                    f"Edge-Kelly identity failed for {pat}: {(m_G - pm) * s_val} != {card_sum_e}"
                )

        # Vertex-Kelly: (n - pn) * ind(F, G) == sum_{v in V(G)} ind(F, G - v) when pn < n
        ind_val = count_induced_subgraphs(n, adj_G, pat)
        cl_ind = parse_exact_int(claimed_kv[pat], f"claimed_kelly_vertex_counts_G[{pat}]")
        if ind_val != cl_ind:
            raise CertificateError(f"Kelly vertex count mismatch for {pat}: claimed {cl_ind}, actual {ind_val}")
        if pn < n:
            card_sum_v = sum(
                count_induced_subgraphs(nc, cadj, pat)
                for (nc, cadj) in vd_G
            )
            if (n - pn) * ind_val != card_sum_v:
                raise CertificateError(
                    f"Vertex-Kelly identity failed for {pat}: {(n - pn) * ind_val} != {card_sum_v}"
                )
        kelly_summary[pat] = {"s_G": s_val, "ind_G": ind_val}

    # Exact Müller inequality: 2^(m-1) > n! (when m >= 1)
    muller_holds = (m_G >= 1) and ((1 << (m_G - 1)) > factorial(n))
    if muller_holds != claimed_muller:
        raise CertificateError(f"claimed_muller_holds mismatch: claimed {claimed_muller}, actual {muller_holds}")

    return {
        "n": n,
        "m_G": m_G,
        "m_H": m_H,
        "degree_seq_G": degree_sequence(adj_G),
        "degree_seq_H": degree_sequence(adj_H),
        "isomorphic": iso_exists,
        "edge_decks_equal": actual_ed,
        "vertex_decks_equal": actual_vd,
        "aut_G": aut_G,
        "aut_H": aut_H,
        "muller_holds": muller_holds,
        "kelly_summary": kelly_summary,
    }


# ---------------------------------------------------------------------------
# Mode 2: kruskal_multiscale_rigidity_cert (Sec. 3 & Lemma 6.1(1))
# ---------------------------------------------------------------------------


def _verify_mode2(payload: Dict[str, Any]) -> Dict[str, Any]:
    _require_keys(
        payload,
        required={
            "n",
            "edges",
            "vertex_order",
            "thresholds",
            "claimed_mst_edges",
            "claimed_scale_summaries",
            "claimed_aut_product_bound",
            "claimed_phi_delta_check_scale",
        },
        optional={"claimed_exact_aut_order", "note"},
        ctx="mode2.payload",
    )

    n = parse_exact_int(payload["n"], "n")
    if n < 4 or n > 64:
        raise CertificateError(f"Mode 2 requires 4 <= n <= 64, got n={n}")
    _, edges = validate_simple_graph(n, payload["edges"], "edges")
    adj = build_adj(n, edges)
    if not is_connected(n, adj):
        raise CertificateError("Mode 2 requires a connected simple graph G")

    vorder_raw = payload["vertex_order"]
    if not isinstance(vorder_raw, list):
        raise CertificateError("vertex_order must be a list")
    vorder = [parse_exact_int(x, f"vertex_order[{i}]") for i, x in enumerate(vorder_raw)]
    try:
        mst_edges, rank = kruskal_degree_ordered_mst(n, edges, vorder)
    except ValueError as exc:
        raise CertificateError(str(exc)) from exc

    # Verify claimed MST matches exact unique lexicographic degree-ordered Kruskal MST
    _, claimed_mst = validate_simple_graph(n, payload["claimed_mst_edges"], "claimed_mst_edges")
    if claimed_mst != mst_edges:
        raise CertificateError(
            f"claimed_mst_edges does not match unique lexicographic degree-ordered Kruskal MST"
        )

    # Verify thresholds: 1 <= Delta_0 < Delta_1 < ... < Delta_L with Delta_L > Delta(G)
    thresh_raw = payload["thresholds"]
    if not isinstance(thresh_raw, list) or len(thresh_raw) < 2:
        raise CertificateError("thresholds must be a list of length >= 2")
    thresholds = [parse_exact_int(x, f"thresholds[{i}]") for i, x in enumerate(thresh_raw)]
    if thresholds[0] < 1:
        raise CertificateError(f"thresholds[0] must be >= 1, got {thresholds[0]}")
    for i in range(len(thresholds) - 1):
        if thresholds[i] >= thresholds[i + 1]:
            raise CertificateError(f"thresholds must be strictly increasing at index {i}")
    max_deg = max(degrees(adj))
    if thresholds[-1] <= max_deg:
        raise CertificateError(f"Final threshold Delta_L={thresholds[-1]} must exceed Delta(G)={max_deg}")

    # Verify root r0 = vertex_order[-1] achieves Delta(G)
    if len(adj[vorder[-1]]) != max_deg:
        raise CertificateError("Root vertex r0 = max_{<_V} V(G) must have degree Delta(G)")

    # Verify each dyadic scale k = 0, ..., L - 1
    claimed_scales = payload["claimed_scale_summaries"]
    if not isinstance(claimed_scales, list) or len(claimed_scales) != len(thresholds) - 1:
        raise CertificateError("claimed_scale_summaries length must equal len(thresholds) - 1")

    total_aut_prod = 1
    verified_scales: List[Dict[str, Any]] = []
    for k in range(len(thresholds) - 1):
        cs = claimed_scales[k]
        if not isinstance(cs, dict):
            raise CertificateError(f"claimed_scale_summaries[{k}] must be an object")
        _require_keys(
            cs,
            required={
                "delta_k",
                "H_k_size",
                "bridge_count",
                "core_tree_edges",
                "class_multiplicities",
                "aut_subgroup_factor",
            },
            optional=set(),
            ctx=f"claimed_scale_summaries[{k}]",
        )
        dk = thresholds[k]
        dk1 = thresholds[k + 1]
        res = analyze_kruskal_scale(n, adj, mst_edges, rank, dk, dk1)

        if not res["spanning_forest_ok"]:
            raise CertificateError(f"Scale k={k} (Delta={dk}): Lemma 3.1(1a) spanning-forest property failed")
        if not res["bridge_bound_ok"]:
            raise CertificateError(f"Scale k={k} (Delta={dk}): Lemma 3.1(1b) bridge bound |B_Delta|<=|H|-1 failed")
        if not res["core_tree_edge_bound_ok"]:
            raise CertificateError(f"Scale k={k} (Delta={dk}): Lemma 3.1(1b) core-tree edge bound failed")
        if not res["canonical_parent_all_ok"]:
            raise CertificateError(f"Scale k={k} (Delta={dk}): Lemma 3.1(1c) canonical T-parent property failed")

        if parse_exact_int(cs["delta_k"], f"scale[{k}].delta_k") != dk:
            raise CertificateError(f"scale[{k}].delta_k mismatch")
        if parse_exact_int(cs["H_k_size"], f"scale[{k}].H_k_size") != res["H_k_size"]:
            raise CertificateError(f"scale[{k}].H_k_size mismatch: actual {res['H_k_size']}")
        if parse_exact_int(cs["bridge_count"], f"scale[{k}].bridge_count") != res["bridge_count"]:
            raise CertificateError(f"scale[{k}].bridge_count mismatch: actual {res['bridge_count']}")
        if parse_exact_int(cs["core_tree_edges"], f"scale[{k}].core_tree_edges") != res["core_tree_edges"]:
            raise CertificateError(f"scale[{k}].core_tree_edges mismatch: actual {res['core_tree_edges']}")
        cl_mults_raw = cs["class_multiplicities"]
        if not isinstance(cl_mults_raw, list):
            raise CertificateError(f"scale[{k}].class_multiplicities must be a list")
        cl_mults = [parse_exact_int(x, f"scale[{k}].class_multiplicities[{j}]") for j, x in enumerate(cl_mults_raw)]
        if cl_mults != res["class_multiplicities"]:
            raise CertificateError(
                f"scale[{k}].class_multiplicities mismatch: claimed {cl_mults}, actual {res['class_multiplicities']}"
            )
        cl_af = parse_exact_int(cs["aut_subgroup_factor"], f"scale[{k}].aut_subgroup_factor")
        if cl_af != res["aut_subgroup_factor"]:
            raise CertificateError(f"scale[{k}].aut_subgroup_factor mismatch: actual {res['aut_subgroup_factor']}")
        total_aut_prod *= int(res["aut_subgroup_factor"])
        verified_scales.append(res)

    claimed_aut_prod = parse_exact_int(payload["claimed_aut_product_bound"], "claimed_aut_product_bound")
    if claimed_aut_prod != total_aut_prod:
        raise CertificateError(
            f"claimed_aut_product_bound mismatch: claimed {claimed_aut_prod}, actual {total_aut_prod}"
        )

    exact_aut: int | None = None
    if "claimed_exact_aut_order" in payload:
        claimed_exact_aut = parse_exact_int(payload["claimed_exact_aut_order"], "claimed_exact_aut_order")
        if n > 14:
            raise CertificateError("claimed_exact_aut_order verification only supported for n <= 14")
        exact_aut = automorphism_order(n, adj)
        if exact_aut != claimed_exact_aut:
            raise CertificateError(
                f"claimed_exact_aut_order mismatch: claimed {claimed_exact_aut}, actual {exact_aut}"
            )
        if total_aut_prod > exact_aut or exact_aut % total_aut_prod != 0:
            raise CertificateError(
                f"Lemma 3.1(2) violated: subgroup product {total_aut_prod} does not divide |Aut(G)|={exact_aut}"
            )

    # Verify Lemma 6.1(1) gap-free internal-edge functional Phi_Delta at claimed_phi_delta_check_scale
    phi_delta = parse_exact_int(payload["claimed_phi_delta_check_scale"], "claimed_phi_delta_check_scale")
    if phi_delta < 1:
        raise CertificateError("claimed_phi_delta_check_scale must be >= 1")
    degs = degrees(adj)
    A_set = {v for v in range(n) if degs[v] >= phi_delta}
    phi_base = phi_delta_functional(adj, phi_delta)
    internal_edges_count = 0
    for idx, (u, v) in enumerate(edges):
        sub_adj = build_adj(n, [e for j, e in enumerate(edges) if j != idx])
        drop = phi_base - phi_delta_functional(sub_adj, phi_delta)
        expected_drop = (1 if u in A_set else 0) + (1 if v in A_set else 0)
        if drop != expected_drop:
            raise CertificateError(
                f"Lemma 6.1(1) Phi_Delta drop mismatch on edge ({u},{v}): got {drop}, expected {expected_drop}"
            )
        if drop == 0:
            internal_edges_count += 1

    return {
        "n": n,
        "m": len(edges),
        "max_deg": max_deg,
        "mst_edge_count": len(mst_edges),
        "scales_verified": len(verified_scales),
        "aut_product_bound": total_aut_prod,
        "exact_aut_order": exact_aut,
        "phi_delta_scale": phi_delta,
        "phi_delta_internal_edges": internal_edges_count,
    }


# ---------------------------------------------------------------------------
# Mode 3: shift_invariant_charlier_band_cert (Sec. 4: Lemmas 4.1, 4.3, Thm 4.2)
# ---------------------------------------------------------------------------


def _verify_mode3(payload: Dict[str, Any]) -> Dict[str, Any]:
    _require_keys(
        payload,
        required={
            "num_vars",
            "degree_N0",
            "terms",
            "target_v",
            "t_max",
            "claimed_l1_norm",
            "claimed_target_coeff",
            "claimed_ratio_l1_over_cv",
            "charlier_check",
        },
        optional={"band_isolation", "note"},
        ctx="mode3.payload",
    )

    M = parse_exact_int(payload["num_vars"], "num_vars")
    N0 = parse_exact_int(payload["degree_N0"], "degree_N0")
    t_max = parse_exact_int(payload["t_max"], "t_max")
    if M < 2 or M > 12:
        raise CertificateError(f"Mode 3 requires 2 <= num_vars <= 12, got {M}")
    if N0 < 1 or t_max < 1:
        raise CertificateError(f"Mode 3 requires degree_N0 >= 1 and t_max >= 1")

    try:
        poly = parse_sparse_poly(payload["terms"], M, "terms")
    except ValueError as exc:
        raise CertificateError(str(exc)) from exc

    # Verify exact homogeneity of degree N0
    for exp in poly:
        if sum(exp) != N0:
            raise CertificateError(f"Polynomial not homogeneous of degree {N0}: exponent {exp} has sum {sum(exp)}")

    # Verify exact shift invariance: D_I(P) == 0 over Q
    deriv = poly_directional_derivative(poly)
    if deriv:
        sample_k = next(iter(deriv))
        raise CertificateError(
            f"Polynomial is NOT shift-invariant: D_I(P) has nonzero term {sample_k}: {deriv[sample_k]}"
        )

    # Verify target exponent vector v
    v_raw = payload["target_v"]
    if not isinstance(v_raw, list) or len(v_raw) != M:
        raise CertificateError(f"target_v must be a list of length {M}")
    v = tuple(parse_exact_int(x, f"target_v[{i}]") for i, x in enumerate(v_raw))
    if any(x < 0 for x in v) or sum(v) != N0:
        raise CertificateError(f"target_v {v} must be nonnegative with sum {N0}")
    if max(v) > t_max:
        raise CertificateError(f"target_v max entry {max(v)} exceeds t_max={t_max}")
    if v not in poly:
        raise CertificateError(f"target_v {v} is not in supp(P)")

    cv = poly[v]
    l1 = poly_l1_norm(poly)
    ratio = l1 / abs(cv)

    cl_l1 = parse_exact_fraction(payload["claimed_l1_norm"], "claimed_l1_norm")
    cl_cv = parse_exact_fraction(payload["claimed_target_coeff"], "claimed_target_coeff")
    cl_ratio = parse_exact_fraction(payload["claimed_ratio_l1_over_cv"], "claimed_ratio_l1_over_cv")
    if cl_l1 != l1:
        raise CertificateError(f"claimed_l1_norm mismatch: claimed {cl_l1}, actual {l1}")
    if cl_cv != cv:
        raise CertificateError(f"claimed_target_coeff mismatch: claimed {cl_cv}, actual {cv}")
    if cl_ratio != ratio:
        raise CertificateError(f"claimed_ratio_l1_over_cv mismatch: claimed {cl_ratio}, actual {ratio}")

    # Verify exact Theorem 4.2 lower bound over Q (zero floats!)
    # General bound: ratio^(200 * t_max) >= (t_max + 1)^N0
    if ratio ** (200 * t_max) < Fraction((t_max + 1) ** N0, 1):
        raise CertificateError(
            f"Theorem 4.2 general bound violated: ratio={ratio}, t={t_max}, N0={N0}"
        )
    # Sharp t = 1 bound: ratio^2 * 3^N0 >= 4^N0  (ratio >= (2 / sqrt(3))^N0)
    if t_max == 1:
        if (ratio ** 2) * (3 ** N0) < Fraction(4 ** N0, 1):
            raise CertificateError(
                f"Theorem 4.2(1) sharp t=1 bound (2/sqrt(3))^N0 violated: ratio={ratio}, N0={N0}"
            )

    # Verify Lemma 4.1(1) exact rational Charlier check
    cc = payload["charlier_check"]
    if not isinstance(cc, dict):
        raise CertificateError("charlier_check must be an object")
    _require_keys(
        cc,
        required={"d", "r", "k_max", "claimed_max_abs_I", "claimed_upper_bound"},
        optional=set(),
        ctx="charlier_check",
    )
    cd = parse_exact_int(cc["d"], "charlier_check.d")
    cr = parse_exact_int(cc["r"], "charlier_check.r")
    ck_max = parse_exact_int(cc["k_max"], "charlier_check.k_max")
    if cr < 4 or cd < 1 or 2 * (cd + 1) > cr:
        raise CertificateError(f"Lemma 4.1(1) requires r >= 4 and 1 <= d <= r/2 - 1, got d={cd}, r={cr}")
    if ck_max < cd + 4 or ck_max > 64:
        raise CertificateError(f"charlier_check.k_max must satisfy d+4 <= k_max <= 64, got {ck_max}")

    lam = Fraction(1, 2 * cr)
    upper_bound = Fraction(1, 1) - Fraction(11 * cd, 24 * cr)
    cl_ub = parse_exact_fraction(cc["claimed_upper_bound"], "charlier_check.claimed_upper_bound")
    if cl_ub != upper_bound:
        raise CertificateError(
            f"charlier_check.claimed_upper_bound mismatch: claimed {cl_ub}, actual {upper_bound}"
        )

    max_abs_val = Fraction(0, 1)
    for k_val in range(ck_max + 1):
        val = abs(charlier_scaled_exact(cd, k_val, cr, lam))
        if val > upper_bound:
            raise CertificateError(
                f"Lemma 4.1(1) violated at k={k_val}: |I_{{{cd},{k_val}}}|={val} > {upper_bound}"
            )
        if val > max_abs_val:
            max_abs_val = val
    cl_max_abs = parse_exact_fraction(cc["claimed_max_abs_I"], "charlier_check.claimed_max_abs_I")
    if cl_max_abs != max_abs_val:
        raise CertificateError(
            f"charlier_check.claimed_max_abs_I mismatch: claimed {cl_max_abs}, actual {max_abs_val}"
        )

    # Optional Lemma 4.3 multiplicity-band isolation audit
    band_summary: Dict[str, Any] | None = None
    if "band_isolation" in payload:
        bi = payload["band_isolation"]
        if not isinstance(bi, dict):
            raise CertificateError("band_isolation must be an object")
        _require_keys(
            bi,
            required={"t0", "t1", "claimed_C_const", "claimed_f_mid_degree", "claimed_f_mid_l1"},
            optional=set(),
            ctx="band_isolation",
        )
        t0 = parse_exact_int(bi["t0"], "band_isolation.t0")
        t1 = parse_exact_int(bi["t1"], "band_isolation.t1")
        if not (0 <= t0 <= t1):
            raise CertificateError(f"band_isolation requires 0 <= t0 <= t1, got [{t0}, {t1}]")
        # Require uniform multiplicity partition across supp(P) (hypothesis of Lemma 4.3!)
        v_sorted = sorted(v)
        for u in poly:
            if sorted(u) != v_sorted:
                raise CertificateError(
                    f"Lemma 4.3 hypothesis failed: support exponent {u} does not share multiplicity partition with v={v}"
                )
        bres = lemma43_band_isolation(poly, v, t0, t1)
        if not bres["I_mid"]:
            raise CertificateError("band_isolation requires non-empty I_mid")
        if not bres["lo_independent"]:
            raise CertificateError("Lemma 4.3 failed: H_w(P) retains dependence on I_lo")
        if bres["target_mid_coeff"] != cv:
            raise CertificateError(
                f"Lemma 4.3 failed: target coefficient not preserved ({bres['target_mid_coeff']} != {cv})"
            )
        if bres["f_mid_l1"] > l1:
            raise CertificateError(f"Lemma 4.3 failed: ||F_mid||_1={bres['f_mid_l1']} > ||P||_1={l1}")
        if not bres["f_mid_shift_invariant"]:
            raise CertificateError("Lemma 4.3 failed: F_mid is not shift-invariant (D_{I_mid}(F_mid) != 0)")

        if parse_exact_int(bi["claimed_C_const"], "band_isolation.claimed_C_const") != bres["C_const"]:
            raise CertificateError(f"band_isolation.claimed_C_const mismatch: actual {bres['C_const']}")
        if parse_exact_int(bi["claimed_f_mid_degree"], "band_isolation.claimed_f_mid_degree") != bres["f_mid_degree"]:
            raise CertificateError(f"band_isolation.claimed_f_mid_degree mismatch: actual {bres['f_mid_degree']}")
        cl_fmid_l1 = parse_exact_fraction(bi["claimed_f_mid_l1"], "band_isolation.claimed_f_mid_l1")
        if cl_fmid_l1 != bres["f_mid_l1"]:
            raise CertificateError(f"band_isolation.claimed_f_mid_l1 mismatch: actual {bres['f_mid_l1']}")

        band_summary = {
            "t0": t0,
            "t1": t1,
            "I_lo_size": len(bres["I_lo"]),
            "I_mid_size": len(bres["I_mid"]),
            "I_hi_size": len(bres["I_hi"]),
            "C_const": bres["C_const"],
            "f_mid_degree": bres["f_mid_degree"],
            "f_mid_l1": fmt_frac(bres["f_mid_l1"]),
        }

    return {
        "num_vars": M,
        "degree_N0": N0,
        "t_max": t_max,
        "support_size": len(poly),
        "l1_norm": fmt_frac(l1),
        "target_coeff": fmt_frac(cv),
        "ratio": fmt_frac(ratio),
        "charlier_max_abs_I": fmt_frac(max_abs_val),
        "charlier_upper_bound": fmt_frac(upper_bound),
        "band_isolation": band_summary,
    }


# ---------------------------------------------------------------------------
# Mode 4: bridge_rigidity_and_density_audit (Sec. 1, 5, 8, 9)
# ---------------------------------------------------------------------------


def _verify_mode4(payload: Dict[str, Any]) -> Dict[str, Any]:
    _require_keys(
        payload,
        required={
            "n",
            "edges",
            "delta",
            "comp_D",
            "v0",
            "fixed_S",
            "claimed_W_D_size",
            "claimed_R",
            "claimed_Q_D_order",
            "claimed_Q_D_primes",
            "density_audit",
        },
        optional={"pillar3_slice", "note"},
        ctx="mode4.payload",
    )

    n = parse_exact_int(payload["n"], "n")
    if n < 4 or n > 64:
        raise CertificateError(f"Mode 4 requires 4 <= n <= 64, got n={n}")
    _, edges = validate_simple_graph(n, payload["edges"], "edges")
    adj = build_adj(n, edges)

    delta = parse_exact_int(payload["delta"], "delta")
    if delta < 2:
        raise CertificateError(f"Mode 4 requires delta >= 2, got {delta}")
    comp_raw = payload["comp_D"]
    fixed_raw = payload["fixed_S"]
    if not isinstance(comp_raw, list) or not comp_raw:
        raise CertificateError("comp_D must be a non-empty list of vertices")
    if not isinstance(fixed_raw, list):
        raise CertificateError("fixed_S must be a list of vertices")
    comp_D = [parse_exact_int(x, f"comp_D[{i}]") for i, x in enumerate(comp_raw)]
    if len(set(comp_D)) != len(comp_D):
        raise CertificateError("Duplicate vertices in comp_D")
    fixed_S = [parse_exact_int(x, f"fixed_S[{i}]") for i, x in enumerate(fixed_raw)]
    if len(set(fixed_S)) != len(fixed_S):
        raise CertificateError("Duplicate vertices in fixed_S")
    v0 = parse_exact_int(payload["v0"], "v0")

    try:
        bres = compute_bridge_restriction_group(n, adj, delta, comp_D, v0, fixed_S)
    except ValueError as exc:
        raise CertificateError(str(exc)) from exc

    if not bres["all_primes_lt_delta"]:
        raise CertificateError(f"Lemma 5.1(2) failed: prime >= Delta={delta} divides |Q_D|")
    if not bres["within_16delta_R_bound"]:
        raise CertificateError(
            f"Lemma 5.1(2) failed: |Q_D|={bres['Q_D_order']} > (16*Delta)^R={bres['bound_16delta_R']}"
        )

    if parse_exact_int(payload["claimed_W_D_size"], "claimed_W_D_size") != len(bres["W_D"]):
        raise CertificateError(f"claimed_W_D_size mismatch: actual {len(bres['W_D'])}")
    if parse_exact_int(payload["claimed_R"], "claimed_R") != bres["R"]:
        raise CertificateError(f"claimed_R mismatch: actual {bres['R']}")
    if parse_exact_int(payload["claimed_Q_D_order"], "claimed_Q_D_order") != bres["Q_D_order"]:
        raise CertificateError(f"claimed_Q_D_order mismatch: actual {bres['Q_D_order']}")
    cl_primes_raw = payload["claimed_Q_D_primes"]
    if not isinstance(cl_primes_raw, list):
        raise CertificateError("claimed_Q_D_primes must be a list")
    cl_primes = [parse_exact_int(x, f"claimed_Q_D_primes[{i}]") for i, x in enumerate(cl_primes_raw)]
    if cl_primes != bres["Q_D_prime_divisors"]:
        raise CertificateError(
            f"claimed_Q_D_primes mismatch: claimed {cl_primes}, actual {bres['Q_D_prime_divisors']}"
        )

    # Sub-audit B1: Exact Theorem 1.1 / Corollary 1.2 / Müller / Remark 1.3 regime audit
    da = payload["density_audit"]
    if not isinstance(da, dict):
        raise CertificateError("density_audit must be an object")
    _require_keys(
        da,
        required={"n0", "m0", "claimed_avg_degree", "claimed_thm11_regime", "claimed_muller_exceeds_n_pow_n"},
        optional=set(),
        ctx="density_audit",
    )
    n0 = parse_exact_int(da["n0"], "density_audit.n0")
    m0 = parse_exact_int(da["m0"], "density_audit.m0")
    if n0 < 4:
        raise CertificateError(f"density_audit.n0 must be >= 4, got {n0}")
    max_simple_edges = n0 * (n0 - 1) // 2
    if m0 < 0 or m0 > max_simple_edges:
        raise CertificateError(
            f"density_audit.m0={m0} impossible for simple graph on n0={n0} (max {max_simple_edges})"
        )
    avg_d = Fraction(2 * m0, n0)
    cl_avg_d = parse_exact_fraction(da["claimed_avg_degree"], "density_audit.claimed_avg_degree")
    if cl_avg_d != avg_d:
        raise CertificateError(f"claimed_avg_degree mismatch: claimed {cl_avg_d}, actual {avg_d}")

    if n0 <= 2000:
        actual_regime = "VACUOUS_SIMPLE_GRAPH_CAP"
    elif m0 >= 1000 * n0:
        actual_regime = "ACTIVE_LINEAR_DENSITY"
    else:
        actual_regime = "BELOW_1000N_THRESHOLD"

    cl_regime = _require_str(da["claimed_thm11_regime"], "density_audit.claimed_thm11_regime")
    if cl_regime != actual_regime:
        raise CertificateError(f"claimed_thm11_regime mismatch: claimed {cl_regime}, actual {actual_regime}")

    # Exact integer comparison: does 2^(m0 - 1) > n0^n0 (sufficient for Müller 2^(m0-1) > n0!)?
    # Check cleanly without constructing huge 2^m0 when m0 is massive:
    # m0 - 1 > n0 * bit_length(n0) guarantees 2^(m0-1) > (2^bit_length(n0))^n0 >= n0^n0;
    # and m0 - 1 <= n0 * (bit_length(n0) - 1) guarantees 2^(m0-1) <= n0^n0.
    # For intermediate cases (or n0 <= 200), compute exact integer power!
    cl_mul = _require_bool(da["claimed_muller_exceeds_n_pow_n"], "density_audit.claimed_muller_exceeds_n_pow_n")
    bl = n0.bit_length()
    if m0 == 0:
        actual_mul = False
    elif m0 - 1 > n0 * bl:
        actual_mul = True
    elif m0 - 1 < n0 * (bl - 1):
        actual_mul = False
    else:
        # Exact arbitrary-precision integer comparison (only reached in narrow transition band)
        if n0 > 10_000:
            raise CertificateError("n0 too large for narrow-band exact power comparison")
        actual_mul = (1 << (m0 - 1)) > (n0 ** n0)
    if cl_mul != actual_mul:
        raise CertificateError(
            f"claimed_muller_exceeds_n_pow_n mismatch: claimed {cl_mul}, actual {actual_mul}"
        )

    # Sub-audit B2: Optional Pillar III degree-excess conservation & 1/2-degree preservation (Prop. 8.4)
    p3_summary: Dict[str, Any] | None = None
    if "pillar3_slice" in payload:
        p3 = payload["pillar3_slice"]
        if not isinstance(p3, dict):
            raise CertificateError("pillar3_slice must be an object")
        _require_keys(
            p3,
            required={"delta_prev", "r_leaf", "twin_classes", "selected_class_indices", "claimed_total_excess"},
            optional=set(),
            ctx="pillar3_slice",
        )
        delta_prev = parse_exact_int(p3["delta_prev"], "pillar3_slice.delta_prev")
        r_leaf = parse_exact_int(p3["r_leaf"], "pillar3_slice.r_leaf")
        if r_leaf < 2:
            raise CertificateError(f"pillar3_slice requires r_leaf >= 2, got {r_leaf}")
        if r_leaf >= delta_prev:
            raise CertificateError(
                f"pillar3_slice requires r_leaf < delta_prev (leaves must lie in V_{{<Delta_prev}}), got r_leaf={r_leaf}, delta_prev={delta_prev}"
            )
        tclasses_raw = p3["twin_classes"]
        sel_raw = p3["selected_class_indices"]
        if not isinstance(tclasses_raw, list) or not tclasses_raw:
            raise CertificateError("pillar3_slice.twin_classes must be non-empty list")
        if not isinstance(sel_raw, list) or not sel_raw:
            raise CertificateError("pillar3_slice.selected_class_indices must be non-empty list")

        degs = degrees(adj)
        H_prev = {v for v in range(n) if degs[v] >= delta_prev}
        seen_w: Set[int] = set()
        seen_nbr_sigs: Set[Tuple[int, ...]] = set()
        tclasses: List[List[int]] = []
        for idx, tc in enumerate(tclasses_raw):
            if not isinstance(tc, list) or len(tc) not in (1, 2):
                raise CertificateError(f"pillar3_slice.twin_classes[{idx}]: Pillar III requires twin class size in {{1,2}}")
            verts = [parse_exact_int(x, f"twin_classes[{idx}][{j}]") for j, x in enumerate(tc)]
            if len(set(verts)) != len(verts):
                raise CertificateError(f"Duplicate vertex inside twin_classes[{idx}]")
            for w in verts:
                if not (0 <= w < n) or w in seen_w:
                    raise CertificateError(f"Invalid or overlapping vertex {w} in twin_classes[{idx}]")
                seen_w.add(w)
                if degs[w] != r_leaf:
                    raise CertificateError(f"Leaf vertex {w} has degree {degs[w]} != r_leaf={r_leaf}")
                if not adj[w].issubset(H_prev):
                    raise CertificateError(f"Leaf vertex {w} has neighbor outside V_{{>=Delta_prev}}(G)")
            # Open-neighborhood twin property inside class & distinctness across classes
            ref_nbrs = adj[verts[0]]
            for w in verts[1:]:
                if adj[w] != ref_nbrs:
                    raise CertificateError(f"Vertices in twin_classes[{idx}] do not share identical open neighborhood")
            nbr_sig = tuple(sorted(ref_nbrs))
            if nbr_sig in seen_nbr_sigs:
                raise CertificateError(
                    f"Distinct Pillar III twin classes share identical open neighborhood {nbr_sig}"
                )
            seen_nbr_sigs.add(nbr_sig)
            tclasses.append(verts)

        sel_idx = [parse_exact_int(x, f"selected_class_indices[{j}]") for j, x in enumerate(sel_raw)]
        if len(set(sel_idx)) != len(sel_idx) or any(not (0 <= i < len(tclasses)) for i in sel_idx):
            raise CertificateError("Invalid or duplicate selected_class_indices")
        # Require >= 3/8 of twin classes selected (Prop. 8.4(2))
        if 8 * len(sel_idx) < 3 * len(tclasses):
            raise CertificateError(
                f"Prop. 8.4(2) sampling lower bound violated: {len(sel_idx)}/{len(tclasses)} < 3/8"
            )

        W0 = {w for i in sel_idx for w in tclasses[i]}
        K0_verts = [v for v in range(n) if v not in W0]
        # Verify half-degree preservation in K0 = G - W0
        for x in K0_verts:
            d_k0 = sum(1 for nb in adj[x] if nb not in W0)
            if 2 * d_k0 < degs[x]:
                raise CertificateError(
                    f"Prop. 8.4(2) half-degree preservation failed at vertex {x}: d_K0={d_k0} < d_G/2={degs[x]}/2"
                )
        # Verify exact degree-excess conservation: sum_{u in H_prev} (d_G(u) - d_K0(u)) == r_leaf * |W0|
        total_excess = sum(
            degs[u] - sum(1 for nb in adj[u] if nb not in W0)
            for u in H_prev
            if u not in W0
        )
        expected_excess = r_leaf * len(W0)
        if total_excess != expected_excess:
            raise CertificateError(
                f"Prop. 8.4(2) degree-excess conservation failed: {total_excess} != {expected_excess}"
            )
        cl_excess = parse_exact_int(p3["claimed_total_excess"], "pillar3_slice.claimed_total_excess")
        if cl_excess != total_excess:
            raise CertificateError(
                f"pillar3_slice.claimed_total_excess mismatch: claimed {cl_excess}, actual {total_excess}"
            )
        p3_summary = {
            "r_leaf": r_leaf,
            "num_twin_classes": len(tclasses),
            "selected_classes": len(sel_idx),
            "W0_size": len(W0),
            "total_degree_excess": total_excess,
        }

    return {
        "delta": delta,
        "W_D_size": len(bres["W_D"]),
        "R": bres["R"],
        "Q_D_order": bres["Q_D_order"],
        "Q_D_primes": bres["Q_D_prime_divisors"],
        "bound_16delta_R": bres["bound_16delta_R"],
        "density_n0": n0,
        "density_m0": m0,
        "avg_degree": fmt_frac(avg_d),
        "thm11_regime": actual_regime,
        "muller_exceeds_n_pow_n": actual_mul,
        "pillar3": p3_summary,
    }


# ---------------------------------------------------------------------------
# Unified Entry Point
# ---------------------------------------------------------------------------


def verify_certificate_dict(cert: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(cert, dict):
        raise CertificateError("Certificate must be a JSON object")
    _check_depth_and_types(cert)
    _require_keys(
        cert,
        required={"schema_version", "cert_id", "mode", "payload"},
        optional={"description"},
        ctx="certificate",
    )
    if cert["schema_version"] != SCHEMA_VERSION:
        raise CertificateError(
            f"Unsupported schema_version: {cert['schema_version']!r} (expected {SCHEMA_VERSION!r})"
        )
    if "description" in cert and not isinstance(cert["description"], str):
        raise CertificateError("description must be a string")
    cert_id = _require_str(cert["cert_id"], "cert_id")
    mode = _require_str(cert["mode"], "mode")
    if mode not in VALID_MODES:
        raise CertificateError(f"Unknown certificate mode: {mode!r}")
    payload = cert["payload"]
    if not isinstance(payload, dict):
        raise CertificateError("payload must be a JSON object")
    if "note" in payload and not isinstance(payload["note"], str):
        raise CertificateError("payload.note must be a string")

    try:
        if mode == "edge_deck_hypomorphism_and_kelly":
            summary = _verify_mode1(payload)
        elif mode == "kruskal_multiscale_rigidity_cert":
            summary = _verify_mode2(payload)
        elif mode == "shift_invariant_charlier_band_cert":
            summary = _verify_mode3(payload)
        else:
            summary = _verify_mode4(payload)
    except CertificateError:
        raise
    except ValueError as exc:
        raise CertificateError(str(exc)) from exc

    return {
        "status": "VERIFIED",
        "schema_version": SCHEMA_VERSION,
        "cert_id": cert_id,
        "mode": mode,
        "summary": summary,
    }


def verify_certificate_json(raw: str) -> Dict[str, Any]:
    cert = load_strict_json(raw)
    return verify_certificate_dict(cert)


def verify_certificate_file(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return verify_certificate_json(p.read_text(encoding="utf-8"))
