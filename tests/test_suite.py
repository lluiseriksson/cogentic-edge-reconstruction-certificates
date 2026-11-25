"""Comprehensive 64-test suite for Paper 60 (EDGERECON-COGENTIC-CERT-1.0):
- 1 aggregate replay of all 16 canonical certificates
- 3 raw-JSON zero-float / NaN / duplicate-key parser guards
- 60 parametrized live adversarial forgery attacks (12 schema/parser + 12 per mode),
  each asserting an exact regex match on the intended CertificateError branch.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.edgerecon_certifier import (
    CertificateError,
    verify_certificate_dict,
    verify_certificate_file,
    verify_certificate_json,
)


CERT_DIR = ROOT / "certificates"


def _load(name: str) -> Dict[str, Any]:
    return json.loads((CERT_DIR / f"{name}.cert.json").read_text(encoding="utf-8"))


def _mod(base: Dict[str, Any], fn: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
    c = copy.deepcopy(base)
    fn(c)
    return c


# ---------------------------------------------------------------------------
# 1. Aggregate Replay of All 16 Canonical Certificates
# ---------------------------------------------------------------------------


def test_all_16_canonical_certificates_verify() -> None:
    paths = sorted(CERT_DIR.glob("*.cert.json"))
    assert len(paths) == 16
    modes_seen = set()
    for p in paths:
        res = verify_certificate_file(p)
        assert res["status"] == "VERIFIED"
        modes_seen.add(res["mode"])
    assert len(modes_seen) == 4


# ---------------------------------------------------------------------------
# 2. Raw JSON Parser Zero-Float, NaN, and Duplicate-Key Guards (3 tests)
# ---------------------------------------------------------------------------


def test_raw_json_rejects_ieee754_float() -> None:
    raw = (CERT_DIR / "m1_harary_m2_counterexample.cert.json").read_text(encoding="utf-8")
    tampered = raw.replace('"n": 4', '"n": 4.0', 1)
    with pytest.raises(CertificateError, match="Floating-point constant rejected"):
        verify_certificate_json(tampered)


def test_raw_json_rejects_nan_constant() -> None:
    raw = (CERT_DIR / "m1_harary_m2_counterexample.cert.json").read_text(encoding="utf-8")
    tampered = raw.replace('"n": 4', '"n": NaN', 1)
    with pytest.raises(CertificateError, match="Floating-point constant rejected"):
        verify_certificate_json(tampered)


def test_raw_json_rejects_duplicate_keys() -> None:
    raw = '{"schema_version":"EDGERECON-COGENTIC-CERT-1.0","schema_version":"EDGERECON-COGENTIC-CERT-1.0"}'
    with pytest.raises(CertificateError, match="Duplicate JSON key"):
        verify_certificate_json(raw)


# ---------------------------------------------------------------------------
# 3. 60 Live Adversarial Forgery Tests (Exact Rejection-Branch Assertions)
# ---------------------------------------------------------------------------


def _build_60_forgeries() -> List[Tuple[str, str, Dict[str, Any]]]:
    m1_c2 = _load("m1_harary_m2_counterexample")
    m1_iso = _load("m1_muller_dense_n6_iso_kelly")
    m1_mis = _load("m1_harary_m5_deck_distinguished")
    m2_12 = _load("m2_kruskal_n12_hub_twins_exact_aut")
    m2_14 = _load("m2_kruskal_n14_two_tier_dyadic")
    m3_t1 = _load("m3_shift_t1_multilinear_n2")
    m3_v4 = _load("m3_vandermonde4_lemma43_band")
    m4_d4 = _load("m4_bridge_d4_wreath48_vacuous2000")
    m4_p3 = _load("m4_bridge_d5_pillar3_r2_active2001")

    def _deep_nest(c: Dict[str, Any]) -> None:
        x: Any = 1
        for _ in range(20):
            x = [x]
        c["payload"]["edges_G"] = x

    def _m1_bad_phi(c: Dict[str, Any]) -> None:
        c["payload"]["isomorphism_phi"] = [0, 1, 2, 3, 4, 5]

    def _m1_self_loop(c: Dict[str, Any]) -> None:
        c["payload"]["edges_G"].append([0, 0])

    def _m1_dup_edge(c: Dict[str, Any]) -> None:
        c["payload"]["edges_G"].append([0, 1])

    def _m2_non_monotone_order(c: Dict[str, Any]) -> None:
        # Put high-degree hub 0 at front of vertex_order
        vo = c["payload"]["vertex_order"]
        vo.remove(0)
        vo.insert(0, 0)

    def _m2_forged_mst(c: Dict[str, Any]) -> None:
        # Swap a valid MST edge with a non-tree edge (1, 2) or (0, 2)
        mst = c["payload"]["claimed_mst_edges"]
        # Replace (0, 1) with (1, 2) if not present, else flip another
        if [0, 1] in mst and [1, 2] not in mst:
            mst.remove([0, 1])
            mst.append([1, 2])
        else:
            mst[0] = [0, 11]

    def _m2_disconnected(c: Dict[str, Any]) -> None:
        c["payload"]["edges"] = [e for e in c["payload"]["edges"] if 4 not in e]

    def _m3_break_shift_invariance(c: Dict[str, Any]) -> None:
        c["payload"]["terms"][0]["coeff"] = "2"

    def _m3_non_homogeneous(c: Dict[str, Any]) -> None:
        c["payload"]["terms"][0]["exp"] = [2, 0, 1, 0]

    def _m3_zero_coeff(c: Dict[str, Any]) -> None:
        c["payload"]["terms"][0]["coeff"] = "0"

    def _m3_dup_monomial(c: Dict[str, Any]) -> None:
        c["payload"]["terms"].append(copy.deepcopy(c["payload"]["terms"][0]))

    def _m3_break_partition_l43(c: Dict[str, Any]) -> None:
        # Add a shift-invariant polynomial with different partition? Easier: tamper band t0 > t1 or non-uniform support
        # Use m3_t1 which has uniform partition (1,1,0,0), wait m3_t1 has (1,0,1,0),(1,0,0,1),(0,1,1,0),(0,1,0,1)
        # Instead construct a non-uniform shift-invariant poly: (y0-y1)^2 = y0^2 - 2 y0 y1 + y1^2 has partitions (2,0) and (1,1)!
        c["payload"]["num_vars"] = 2
        c["payload"]["degree_N0"] = 2
        c["payload"]["terms"] = [
            {"exp": [2, 0], "coeff": "1"},
            {"exp": [1, 1], "coeff": "-2"},
            {"exp": [0, 2], "coeff": "1"},
        ]
        c["payload"]["target_v"] = [2, 0]
        c["payload"]["t_max"] = 2
        c["payload"]["claimed_l1_norm"] = "4"
        c["payload"]["claimed_target_coeff"] = "1"
        c["payload"]["claimed_ratio_l1_over_cv"] = "4"
        c["payload"]["band_isolation"] = {
            "t0": 1,
            "t1": 2,
            "claimed_C_const": 2,
            "claimed_f_mid_degree": 1,
            "claimed_f_mid_l1": "2",
        }

    def _m4_non_component_D(c: Dict[str, Any]) -> None:
        c["payload"]["comp_D"] = [6, 7, 8]  # missing vertex 9 of connected component

    def _m4_root_outside_D(c: Dict[str, Any]) -> None:
        c["payload"]["v0"] = 0

    def _m4_bad_twin_class(c: Dict[str, Any]) -> None:
        # Put vertices 11 (nbrs {0,1}) and 13 (nbrs {2,3}) in the same twin class
        c["payload"]["pillar3_slice"]["twin_classes"] = [[11, 13], [12, 14], [15], [16, 17]]

    def _m4_low_sampling_ratio(c: Dict[str, Any]) -> None:
        # 1/4 = 0.25 < 3/8 = 0.375
        c["payload"]["pillar3_slice"]["selected_class_indices"] = [0]

    def _m4_half_deg_violation(c: Dict[str, Any]) -> None:
        # Dropping edges (0,2),(0,3),(0,5) lowers hub 0 from deg 10 to deg 7 (>= 5),
        # and selecting [0, 3] removes 4 leaves adjacent to hub 0 -> d_K0(0) = 3 < 7/2!
        c["payload"]["edges"] = [
            e for e in c["payload"]["edges"]
            if e not in ([0, 2], [0, 3], [0, 5])
        ]
        c["payload"]["pillar3_slice"]["selected_class_indices"] = [0, 3]
        c["payload"]["pillar3_slice"]["claimed_total_excess"] = 8

    def _m4_p3_split_size4_twin_class(c: Dict[str, Any]) -> None:
        # Rewire leaves 13, 14 from {2, 3} to {0, 1}, creating a size-4 twin class {11,12,13,14}
        # masquerading as two size-2 Pillar III classes [[11,12], [13,14]]!
        new_edges = [e for e in c["payload"]["edges"] if 13 not in e and 14 not in e]
        new_edges.extend([[0, 13], [1, 13], [0, 14], [1, 14]])
        c["payload"]["edges"] = sorted(new_edges)

    return [
        # --- 12 Schema & Parser Forgeries ---
        ("S01_wrong_schema", "Unsupported schema_version", _mod(m1_c2, lambda c: c.__setitem__("schema_version", "BAD-1.0"))),
        ("S02_unknown_mode", "Unknown certificate mode", _mod(m1_c2, lambda c: c.__setitem__("mode", "bogus_mode"))),
        ("S03_missing_payload", "missing required keys", _mod(m1_c2, lambda c: c.pop("payload"))),
        ("S04_extra_top_key", "unexpected extra keys", _mod(m1_c2, lambda c: c.__setitem__("extra", 1))),
        ("S05_empty_cert_id", "expected non-empty string", _mod(m1_c2, lambda c: c.__setitem__("cert_id", "   "))),
        ("S06_bool_as_int", "boolean not permitted", _mod(m1_c2, lambda c: c["payload"].__setitem__("n", True))),
        ("S07_float_in_dict", "Floating-point value", _mod(m1_c2, lambda c: c["payload"].__setitem__("n", 4.5))),
        ("S08_unicode_digit_str", "invalid exact integer", _mod(m1_c2, lambda c: c["payload"].__setitem__("n", "٤"))),
        ("S09_decimal_rational_str", "decimal/scientific notation", _mod(m3_t1, lambda c: c["payload"].__setitem__("claimed_l1_norm", "4.0"))),
        ("S10_zero_denominator_frac", "denominator must be positive", _mod(m3_t1, lambda c: c["payload"].__setitem__("claimed_l1_norm", "4/0"))),
        ("S11_excessive_nesting", "nesting depth exceeded", _mod(m1_c2, _deep_nest)),
        ("S12_extra_payload_key", "unexpected extra keys", _mod(m1_c2, lambda c: c["payload"].__setitem__("injected", 0))),

        # --- 12 Mode 1 (Edge-Deck, Hypomorphism & Kelly) Forgeries ---
        ("M1_01_n_too_small", "requires 4 <= n <= 10", _mod(m1_c2, lambda c: c["payload"].__setitem__("n", 3))),
        ("M1_02_self_loop", "self-loop", _mod(m1_c2, _m1_self_loop)),
        ("M1_03_duplicate_edge", "duplicate edge", _mod(m1_c2, _m1_dup_edge)),
        ("M1_04_forged_aut_G", "claimed_aut_G mismatch", _mod(m1_c2, lambda c: c["payload"].__setitem__("claimed_aut_G", 99))),
        ("M1_05_forged_aut_H", "claimed_aut_H mismatch", _mod(m1_c2, lambda c: c["payload"].__setitem__("claimed_aut_H", 99))),
        ("M1_06_forged_edge_deck_flag", "claimed_edge_decks_equal mismatch", _mod(m1_c2, lambda c: c["payload"].__setitem__("claimed_edge_decks_equal", False))),
        ("M1_07_forged_vertex_deck_flag", "claimed_vertex_decks_equal mismatch", _mod(m1_c2, lambda c: c["payload"].__setitem__("claimed_vertex_decks_equal", True))),
        ("M1_08_iso_missing_phi", "requires explicit witness isomorphism_phi", _mod(m1_iso, lambda c: c["payload"].pop("isomorphism_phi"))),
        ("M1_09_iso_bad_phi", "fails exact edge-preservation", _mod(m1_iso, _m1_bad_phi)),
        ("M1_10_false_counterexample_on_iso", "G and H are actually isomorphic", _mod(m1_iso, lambda c: c["payload"].__setitem__("verdict", "NON_ISOMORPHIC_COUNTEREXAMPLE"))),
        ("M1_11_forged_kelly_edge_count", "Kelly edge count mismatch", _mod(m1_iso, lambda c: c["payload"]["claimed_kelly_edge_counts_G"].__setitem__("K3", 999))),
        ("M1_12_forged_muller_flag", "claimed_muller_holds mismatch", _mod(m1_mis, lambda c: c["payload"].__setitem__("claimed_muller_holds", True))),

        # --- 12 Mode 2 (Single Kruskal MST Multi-Scale Rigidity) Forgeries ---
        ("M2_01_disconnected_graph", "connected simple graph", _mod(m2_12, _m2_disconnected)),
        ("M2_02_non_monotone_vorder", "violates degree monotonicity", _mod(m2_12, _m2_non_monotone_order)),
        ("M2_03_forged_mst_edge", "does not match unique lexicographic", _mod(m2_12, _m2_forged_mst)),
        ("M2_04_non_increasing_thresh", "must be strictly increasing", _mod(m2_14, lambda c: c["payload"].__setitem__("thresholds", [3, 3, 9]))),
        ("M2_05_low_final_thresh", "must exceed Delta\\(G\\)", _mod(m2_12, lambda c: c["payload"].__setitem__("thresholds", [3, 5]))),
        ("M2_06_forged_H_k_size", "H_k_size mismatch", _mod(m2_12, lambda c: c["payload"]["claimed_scale_summaries"][0].__setitem__("H_k_size", 4))),
        ("M2_07_forged_bridge_count", "bridge_count mismatch", _mod(m2_12, lambda c: c["payload"]["claimed_scale_summaries"][0].__setitem__("bridge_count", 2))),
        ("M2_08_forged_core_edges", "core_tree_edges mismatch", _mod(m2_12, lambda c: c["payload"]["claimed_scale_summaries"][0].__setitem__("core_tree_edges", 9))),
        ("M2_09_forged_class_mults", "class_multiplicities mismatch", _mod(m2_12, lambda c: c["payload"]["claimed_scale_summaries"][0].__setitem__("class_multiplicities", [4, 1]))),
        ("M2_10_forged_aut_factor", "aut_subgroup_factor mismatch", _mod(m2_12, lambda c: c["payload"]["claimed_scale_summaries"][0].__setitem__("aut_subgroup_factor", 24))),
        ("M2_11_forged_aut_product", "claimed_aut_product_bound mismatch", _mod(m2_12, lambda c: c["payload"].__setitem__("claimed_aut_product_bound", 999))),
        ("M2_12_forged_exact_aut", "claimed_exact_aut_order mismatch", _mod(m2_12, lambda c: c["payload"].__setitem__("claimed_exact_aut_order", 12))),

        # --- 12 Mode 3 (Shift-Invariant Polynomials, Charlier & Band Isolation) Forgeries ---
        ("M3_01_not_shift_invariant", "NOT shift-invariant", _mod(m3_t1, _m3_break_shift_invariance)),
        ("M3_02_not_homogeneous", "not homogeneous", _mod(m3_t1, _m3_non_homogeneous)),
        ("M3_03_zero_coeff_in_supp", "zero coefficient forbidden", _mod(m3_t1, _m3_zero_coeff)),
        ("M3_04_duplicate_monomial", "duplicate exponent vector", _mod(m3_t1, _m3_dup_monomial)),
        ("M3_05_target_not_in_supp", "is not in supp\\(P\\)", _mod(m3_t1, lambda c: c["payload"].__setitem__("target_v", [1, 1, 0, 0]))),
        ("M3_06_target_exceeds_tmax", "exceeds t_max", _mod(m3_v4, lambda c: c["payload"].__setitem__("t_max", 2))),
        ("M3_07_forged_l1_norm", "claimed_l1_norm mismatch", _mod(m3_t1, lambda c: c["payload"].__setitem__("claimed_l1_norm", "5"))),
        ("M3_08_forged_ratio", "claimed_ratio_l1_over_cv mismatch", _mod(m3_t1, lambda c: c["payload"].__setitem__("claimed_ratio_l1_over_cv", "3"))),
        ("M3_09_charlier_d_too_large", "Lemma 4.1\\(1\\) requires", _mod(m3_t1, lambda c: c["payload"]["charlier_check"].__setitem__("d", 3))),
        ("M3_10_forged_charlier_max", "claimed_max_abs_I mismatch", _mod(m3_t1, lambda c: c["payload"]["charlier_check"].__setitem__("claimed_max_abs_I", "1/2"))),
        ("M3_11_non_uniform_partition_l43", "does not share multiplicity partition", _mod(m3_t1, _m3_break_partition_l43)),
        ("M3_12_forged_band_C_const", "claimed_C_const mismatch", _mod(m3_v4, lambda c: c["payload"]["band_isolation"].__setitem__("claimed_C_const", 12))),

        # --- 12 Mode 4 (Bridge Rigidity, Pillar III Conservation & Density Regimes) Forgeries ---
        ("M4_01_non_component_D", "not a connected component", _mod(m4_d4, _m4_non_component_D)),
        ("M4_02_root_outside_D", "must lie in comp_D", _mod(m4_d4, _m4_root_outside_D)),
        ("M4_03_forged_WD_size", "claimed_W_D_size mismatch", _mod(m4_d4, lambda c: c["payload"].__setitem__("claimed_W_D_size", 5))),
        ("M4_04_forged_R", "claimed_R mismatch", _mod(m4_d4, lambda c: c["payload"].__setitem__("claimed_R", 4))),
        ("M4_05_forged_QD_order", "claimed_Q_D_order mismatch", _mod(m4_d4, lambda c: c["payload"].__setitem__("claimed_Q_D_order", 24))),
        ("M4_06_forged_QD_primes", "claimed_Q_D_primes mismatch", _mod(m4_d4, lambda c: c["payload"].__setitem__("claimed_Q_D_primes", [2, 3, 5]))),
        ("M4_07_impossible_simple_m0", "impossible for simple graph", _mod(m4_d4, lambda c: c["payload"]["density_audit"].__setitem__("m0", 2_000_000))),
        ("M4_08_forged_thm11_regime", "claimed_thm11_regime mismatch", _mod(m4_d4, lambda c: c["payload"]["density_audit"].__setitem__("claimed_thm11_regime", "ACTIVE_LINEAR_DENSITY"))),
        ("M4_09_forged_muller_pow_flag", "claimed_muller_exceeds_n_pow_n mismatch", _mod(m4_d4, lambda c: c["payload"]["density_audit"].__setitem__("claimed_muller_exceeds_n_pow_n", False))),
        ("M4_10_p3_non_twins_in_class", "do not share identical open neighborhood", _mod(m4_p3, _m4_bad_twin_class)),
        ("M4_11_p3_below_3_8_sampling", "sampling lower bound violated", _mod(m4_p3, _m4_low_sampling_ratio)),
        ("M4_12_p3_half_degree_break", "half-degree preservation failed", _mod(m4_p3, _m4_half_deg_violation)),
        ("M4_13_p3_rleaf_ge_delta_prev", "requires r_leaf < delta_prev", _mod(m4_p3, lambda c: c["payload"]["pillar3_slice"].__setitem__("delta_prev", 2))),
        ("M4_14_p3_split_size4_twin_class", "Distinct Pillar III twin classes share identical open neighborhood", _mod(m4_p3, _m4_p3_split_size4_twin_class)),
    ]


@pytest.mark.parametrize("attack_id,expected_regex,forged_cert", _build_60_forgeries())
def test_adversarial_forgery_rejected(
    attack_id: str,
    expected_regex: str,
    forged_cert: Dict[str, Any],
) -> None:
    with pytest.raises(CertificateError, match=expected_regex):
        verify_certificate_dict(forged_cert)
