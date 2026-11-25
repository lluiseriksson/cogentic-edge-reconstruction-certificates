# Constructive Lean 4 Kernel and Exact Zero-Float Certificates for Google Research Cogentic's Linear-Density Edge Reconstruction Theorem

**Author:** Lluis Eriksson (Independent Researcher)
**License:** MIT (code, Lean 4 kernel, certificates) / CC-BY-4.0 (manuscript & figures)
**Target Breakthrough:** Google Research Cogentic (Nov. 25, 2026), *Unconditional Edge Reconstruction of Simple Graphs of Linear Density* ($m \ge 1000\,n_0$, $\delta(G_0) \ge 2000$).

## Repository Architecture
- `lean/CogenticEdgeRecon.lean` + `lean/AuditAxioms.lean`: 18 standalone constructive Lean 4.35.0 theorems (0 `sorry`, 0 `axiom`, 0 `Classical.choice`, 0 warnings; 5 zero-axiom `none` + 13 `[propext, Quot.sound]`).
- `src/edgerecon_exact.py` + `src/edgerecon_certifier.py`: Strictly zero-float (`int` / `fractions.Fraction`) 4-mode verifier (`EDGERECON-COGENTIC-CERT-1.0`).
- `certificates/`: 16 canonical exact certificates across all 4 modes (Harary sharp $m \le 3$ pairs, Kelly/Greenwell/Müller audits, multi-scale Kruskal MSTs, exact shift-invariant Charlier/Vandermonde band isolations, wreath-product bridge restriction groups, Pillar III $2/5$-sampling conservation, and $n_0 = 2^{1005}$ beating Müller).
- `tests/test_suite.py`: 66 pytest checks (16-cert replay + 3 raw-JSON parser guards + 62 live adversarial forgery attacks with exact rejection-branch assertions).
- `scripts/verify_all.py`: Unified 5-gate verifier (Lean build/axioms, zero-float AST audit, 16-cert replay, 66/66 pytest, 40-file SHA-256 manifest).
- `paper/main.pdf`: 4-page publication manuscript with 3 300-DPI figures and worked hand-verifiable witnesses.

## Quick Verification
```bash
python scripts/verify_all.py
```
