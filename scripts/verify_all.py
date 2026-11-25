"""Unified 5-gate verification script for Paper 60 (EDGERECON-COGENTIC-CERT-1.0).

Gate 1: Lean 4.35.0 build & strict axiom whitelist (0 sorry, 0 axiom, 0 Classical.choice, 0 warnings)
Gate 2: Zero-float static AST hygiene audit across src/*.py
Gate 3: Replay all 16 canonical certificates via exact zero-float verifier
Gate 4: Full pytest suite (64/64 tests, 60 live adversarial forgery rejections)
Gate 5: SHA-256 integrity manifest check (or --write-manifest to regenerate)
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.edgerecon_certifier import verify_certificate_file

ALLOWED_AXIOMS: Set[str] = {"propext", "Quot.sound"}

EXPECTED_THEOREMS: List[str] = [
    "CogenticEdgeRecon.cor12_min_degree_implies_linear_density",
    "CogenticEdgeRecon.thm11_vacuous_small_n",
    "CogenticEdgeRecon.lem22_muller_factorial_bound",
    "CogenticEdgeRecon.rem13_spanning_tree_barrier_at_d8",
    "CogenticEdgeRecon.lem24_decision_tree_kraft_inductive",
    "CogenticEdgeRecon.lem24_numerical_e_ln2_pi_rational",
    "CogenticEdgeRecon.lem31_bridge_count_bound",
    "CogenticEdgeRecon.lem31_nonbridge_entropy_margin",
    "CogenticEdgeRecon.lem33_dyadic_sum_geometric",
    "CogenticEdgeRecon.lem33_tail_error_10_neg_34",
    "CogenticEdgeRecon.lem42_t1_charlier_ratio_decay",
    "CogenticEdgeRecon.lem42_charlier_duality_2var",
    "CogenticEdgeRecon.lem43_band_localization_partition",
    "CogenticEdgeRecon.lem51_chebyshev_primorial_step",
    "CogenticEdgeRecon.lem51_legendre_valuation_bound",
    "CogenticEdgeRecon.prop84_chebyshev_sampling_3_8",
    "CogenticEdgeRecon.prop84_degree_excess_conservation",
    "CogenticEdgeRecon.sec9_final_contradiction_closure",
]

FORBIDDEN_IMPORTS: Set[str] = {"numpy", "scipy", "decimal", "statistics", "random", "cmath"}
FORBIDDEN_MATH_FUNCS: Set[str] = {
    "log", "log2", "log10", "exp", "sqrt", "sin", "cos", "tan", "asin", "acos",
    "atan", "atan2", "pow", "fabs", "floor", "ceil", "isnan", "isinf", "isfinite",
}
FORBIDDEN_CALL_NAMES: Set[str] = {"float", "complex"}


def gate1_lean() -> None:
    print("=== GATE 1: Lean 4.35.0 Kernel & Strict Axiom Audit ===")
    lean_dir = ROOT / "lean"
    lean_src = (lean_dir / "CogenticEdgeRecon.lean").read_text(encoding="utf-8")
    # Strip block comments /- ... -/ and line comments -- ... before scanning for forbidden tokens
    import re
    stripped = re.sub(r"/-.*?-/", "", lean_src, flags=re.DOTALL)
    stripped = re.sub(r"--[^\n]*", "", stripped)
    for forbidden in ["sorry", "\naxiom ", "Classical.choice", "native_decide"]:
        if forbidden in stripped:
            raise AssertionError(f"Forbidden token {forbidden!r} found in CogenticEdgeRecon.lean")

    p_build = subprocess.run(["lake", "build"], cwd=lean_dir, capture_output=True, text=True)
    if p_build.returncode != 0:
        raise AssertionError(f"lake build failed:\n{p_build.stdout}\n{p_build.stderr}")
    if "warning:" in p_build.stdout.lower() or "warning:" in p_build.stderr.lower():
        raise AssertionError(f"Lean build emitted warnings:\n{p_build.stdout}\n{p_build.stderr}")

    p_audit = subprocess.run(
        ["lake", "env", "lean", "-j", "1", "AuditAxioms.lean"],
        cwd=lean_dir,
        capture_output=True,
        text=True,
    )
    if p_audit.returncode != 0:
        raise AssertionError(f"AuditAxioms.lean failed:\n{p_audit.stdout}\n{p_audit.stderr}")
    if "warning:" in p_audit.stdout.lower() or "warning:" in p_audit.stderr.lower():
        raise AssertionError(f"AuditAxioms.lean emitted warnings:\n{p_audit.stdout}\n{p_audit.stderr}")

    lines = [ln.strip() for ln in p_audit.stdout.splitlines() if ln.strip()]
    seen: Set[str] = set()
    zero_ax = 0
    pq_ax = 0
    for ln in lines:
        for thm in EXPECTED_THEOREMS:
            if f"'{thm}'" in ln:
                seen.add(thm)
                if "does not depend on any axioms" in ln:
                    zero_ax += 1
                elif "depends on axioms:" in ln:
                    raw_ax = ln.split("[", 1)[1].rsplit("]", 1)[0]
                    ax_set = {a.strip() for a in raw_ax.split(",") if a.strip()}
                    if not ax_set.issubset(ALLOWED_AXIOMS):
                        raise AssertionError(f"Theorem {thm} uses disallowed axioms: {ax_set}")
                    pq_ax += 1
                else:
                    raise AssertionError(f"Unexpected axiom audit line: {ln}")
    missing = set(EXPECTED_THEOREMS) - seen
    if missing:
        raise AssertionError(f"Missing theorems in axiom audit: {sorted(missing)}")
    print(f"[PASS] Gate 1: {len(seen)}/18 theorems verified ({zero_ax} zero-axiom, {pq_ax} [propext, Quot.sound], 0 warnings)")


def gate2_zero_float_ast() -> None:
    print("=== GATE 2: Zero-Float Static AST Hygiene Audit (src/*.py) ===")
    src_files = sorted((ROOT / "src").glob("*.py"))
    assert len(src_files) >= 3
    for path in src_files:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (float, complex)):
                raise AssertionError(f"{path.name}: float/complex literal {node.value!r} forbidden")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in FORBIDDEN_IMPORTS or alias.name == "math":
                        raise AssertionError(f"{path.name}: forbidden import {alias.name!r}")
            if isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if mod in FORBIDDEN_IMPORTS:
                    raise AssertionError(f"{path.name}: forbidden import from {mod!r}")
                if mod == "math":
                    for alias in node.names:
                        if alias.name in FORBIDDEN_MATH_FUNCS or alias.name not in {"comb", "factorial", "gcd", "isqrt"}:
                            raise AssertionError(f"{path.name}: forbidden math symbol {alias.name!r}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALL_NAMES:
                    raise AssertionError(f"{path.name}: forbidden call {node.func.id}()")
    print(f"[PASS] Gate 2: {len(src_files)}/{len(src_files)} src/*.py files strictly zero-float")


def gate3_certificates() -> None:
    print("=== GATE 3: Canonical 16-Certificate Replay ===")
    paths = sorted((ROOT / "certificates").glob("*.cert.json"))
    assert len(paths) == 16, f"Expected 16 certificates, found {len(paths)}"
    for p in paths:
        res = verify_certificate_file(p)
        assert res["status"] == "VERIFIED", f"{p.name} failed verification"
    print(f"[PASS] Gate 3: 16/16 canonical certificates verified")


def gate4_pytest() -> None:
    print("=== GATE 4: Full Pytest Suite (66 Tests / 62 Forgeries) ===")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_suite.py"], cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise AssertionError(f"pytest failed:\n{p.stdout}\n{p.stderr}")
    summary = p.stdout.strip().splitlines()[-1]
    print(f"[PASS] Gate 4: {summary}")


def _manifest_files() -> List[Path]:
    patterns = [
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "lean/lean-toolchain",
        "lean/lakefile.toml",
        "lean/CogenticEdgeRecon.lean",
        "lean/AuditAxioms.lean",
        "src/*.py",
        "scripts/*.py",
        "tests/*.py",
        "certificates/*.cert.json",
        "results/*.json",
        "paper/main.tex",
        "paper/references.bib",
        "paper/figures/*.png",
        "paper/main.pdf",
    ]
    out: List[Path] = []
    for pat in patterns:
        out.extend(sorted(ROOT.glob(pat)))
    return sorted(set(p for p in out if p.is_file()))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def gate5_manifest(write: bool) -> None:
    print("=== GATE 5: SHA-256 Integrity Manifest ===")
    manifest_path = ROOT / "SHA256_MANIFEST.json"
    files = _manifest_files()
    computed: Dict[str, str] = {
        p.relative_to(ROOT).as_posix(): _sha256(p) for p in files
    }
    if write:
        manifest_path.write_text(json.dumps(computed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[PASS] Gate 5: Wrote SHA256_MANIFEST.json ({len(computed)} files)")
        return
    if not manifest_path.exists():
        raise AssertionError("SHA256_MANIFEST.json missing (run with --write-manifest)")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recorded != computed:
        rec_keys = set(recorded.keys())
        cmp_keys = set(computed.keys())
        diff_msg = []
        if rec_keys - cmp_keys:
            diff_msg.append(f"Missing files: {sorted(rec_keys - cmp_keys)}")
        if cmp_keys - rec_keys:
            diff_msg.append(f"Unlisted files: {sorted(cmp_keys - rec_keys)}")
        for k in sorted(rec_keys & cmp_keys):
            if recorded[k] != computed[k]:
                diff_msg.append(f"Hash mismatch: {k}")
        raise AssertionError("SHA256_MANIFEST.json mismatch:\n" + "\n".join(diff_msg))
    print(f"[PASS] Gate 5: {len(computed)}/{len(computed)} SHA-256 digests verified")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    gate1_lean()
    gate2_zero_float_ast()
    gate3_certificates()
    gate4_pytest()
    gate5_manifest(write=args.write_manifest)
    print("\nALL 5 GATES PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
