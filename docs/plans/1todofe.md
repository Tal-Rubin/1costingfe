# Critical Analysis: 1costingfe vs. fusion-tea Plan

## The Strategic Question

The fusion-tea plan (2026-02-17) describes a **three-layer hybrid architecture** where SysML owns structure/accounting, a physics engine validates feasibility, and PyFECONS computes component costs. The plan's Phase 2 calls for a thin **adapter layer** that maps SysML parameters → PyFECONS input classes → `RunCosting()` → component costs back.

1costingfe is **not that adapter**. It's a ground-up replacement for PyFECONS itself — a standalone JAX-native costing engine. This is a much bigger bet, with both upside and risk.

---

## What 1costingfe Gets Right

**1. Customer-first inverse power balance** — The plan (Decision #7) says "customer specifies p_net; adapter derives p_fusion via inverse power balance." 1costingfe does exactly this, and makes it algebraic (closed-form) rather than iterative. This is better than PyFECONS.

**2. JAX differentiability** — Exact gradients via `jax.grad` replace PyFECONS's finite-difference sensitivity analysis. This is a genuine architectural improvement: faster, more accurate, and enables `jax.vmap` for batch sweeps.

**3. Multi-fuel parity** — All 4 fuels (DT, DD, DHe3, pB11) are first-class citizens throughout: ash/neutron splits, blanket sizing, shielding, licensing, O&M. The plan's Decision #9 (feedstock in CAS80, breeding capital in CAS22) is implemented.

**4. Clean functional architecture** — Pure functions, frozen dataclasses, explicit `CostingConstants`. No global state. This is more testable and composable than PyFECONS's class-based structure.

**5. CAS coverage** — Full CAS 10–90 with CAS22 broken into 12+ sub-accounts. The plan's Phase 3 financial formulas (CAS 10, 30–60, 70–90, LCOE) are all present.

**6. Adapter exists** — `adapter.py` provides `run_costing(FusionTeaInput) → FusionTeaOutput` with CAS-code-keyed results, which is the interface shape the plan needs.

---

## What's Missing or Divergent

### A. Plan-Critical Gaps

| Plan Requirement | PyFECONS Has It | 1costingfe Status |
|---|---|---|
| Input validation (460 lines of range checks, cross-field physics) | Yes | **Missing entirely** — zero validation |
| Physics feasibility checks (density, wall loading, heat flux, confinement time) | Yes | **Missing entirely** |
| Costing sanity warnings (Q_sci < 1, p_net ≤ 0, heating mismatch) | Yes | **Missing entirely** |
| LSA factor system (Level of Safety Assurance → indirect cost scaling) | Yes | **Missing** — CAS30 uses power scaling, not LSA |
| DPA / neutron damage model | Yes | **Missing** — no FW lifetime → replacement chain |
| Historical scaling / learning curves | Yes | **Missing** |
| Cost interpolation (4 range bounds × optimism × learning) | Plan Decision #8 | **Missing** — single point estimates only |
| ECRH/LHCD sub-account (CAS 22.01.04) | Yes | **Missing** — heating is one lump sum |
| Cryoplant (CAS 22.03.02) | Yes | **Missing** |
| Pellet injection (CAS 22.05.08) | Yes | **Missing** |
| I&C scaling with p_th | Yes | **Missing** |

The zero-validation issue is the most concerning. PyFECONS has 460 lines of input validation + physics feasibility checks that catch garbage-in-garbage-out scenarios. 1costingfe will silently produce LCOE for physically impossible designs.

### B. Boundary Violations

The plan explicitly draws a boundary:

> "SysML owns structure and accounting. PyFECONS owns component costing."
> "SysML-side Accounting (CAS aggregation, CAS 10/30-90 financial calcs, LCOE)"

1costingfe computes **everything** — CAS aggregation, CAS 10/30–90 financials, and LCOE — internally. This means:

- SysML's financial CalcDefs (proven on solar_battery) become redundant
- The pipeline can't mix-and-match: you can't override CAS30 logic in SysML without forking 1costingfe
- The plan's three-layer separation collapses into two layers

This may be intentional (simpler pipeline), but it contradicts the plan's architecture.

### C. SysML Integration Gap

The plan's pipeline is:
```
SysML → extract params (syside AST) → Physics Engine → Adapter → SysML accounting → LCOE
```

1costingfe's pipeline is:
```
FusionTeaInput (flat dict) → CostModel.forward() → FusionTeaOutput (flat dict)
```

Missing pieces for fusion-tea integration:
- No SysML AST traversal / parameter extraction
- No `CostEvaluatorResult` type (the solar_battery pattern)
- No TEAx module wrapper
- No mechanism for SysML to inject per-component costs back into its tree
- `FusionTeaInput.overrides` is a flat dict — no hierarchy matching the SysML part tree

### D. CAS22 Simplifications

1costingfe's CAS22 uses a hybrid `volume × (power/power_ref)^0.6` scaling. This is reasonable but coarser than PyFECONS which has:
- Separate TF/CS/PF/shim coil costing with `kAm` calculations (the plan's Decision #3: `total_kAm = G × B × R² / (μ₀ × 1000)`)
- Material-specific costs (HTS vs LTS vs copper)
- The simplified coil interface (`b_max`, `r_coil`, `coil_material`) exists in PyFECONS but not in 1costingfe

1costingfe's `coils_base = 500.0 M$` with power scaling is a rougher approximation.

---

## Architectural Assessment

**The fundamental tension:** The plan designed a thin adapter wrapping PyFECONS. 1costingfe is a thick replacement of PyFECONS. These are different strategies:

| | Thin Adapter (plan) | Thick Replacement (1costingfe) |
|---|---|---|
| Reuse of PyFECONS validation | Full | None |
| JAX differentiability | No (PyFECONS is NumPy) | Yes |
| Development effort | Low | High |
| Maintenance burden | Two codebases | One codebase |
| Feature parity risk | Zero | **High** (current gaps) |

The thick replacement is the higher-payoff bet if you can close the gaps. The JAX integration alone is worth it — finite-difference sensitivity on a 25-parameter model is fragile and slow. But right now, 1costingfe is roughly at **60–70% feature parity** with PyFECONS's costing capabilities, and at **0% parity** on validation/warnings.

---

## Recommendations (Priority Order)

1. **Add input validation and warnings** — This is the single biggest gap. Port PyFECONS's validation logic (range checks, physics feasibility, costing sanity). Without this, 1costingfe is a calculator that happily computes LCOE for a plasma with negative temperature.

2. **Decide the boundary** — Either:
   - (a) 1costingfe computes everything (current approach) and fusion-tea's SysML financial CalcDefs are deprecated, OR
   - (b) 1costingfe returns only component costs (CAS 22.xx), and SysML handles CAS 10/30–90/LCOE.

   The plan says (b). The code does (a). Pick one and commit.

3. **Add LSA factor system** — The plan specifies LSA 2 as default for CATF MFE. 1costingfe's CAS30 indirect costs use a power-scaling formula that doesn't account for safety assurance level. This affects cost by 10–20%.

4. **Break up CAS22 coil costing** — Replace the lump `coils_base = 500 M$` with the simplified coil model the plan already designed (`b_max`, `r_coil`, `coil_material` → `kAm` → cost). PyFECONS has this; it's the highest-leverage CAS22 improvement.

5. **Add cost uncertainty bands** — The plan's Decision #8 calls for 4 range bounds × (optimism, learning) bilinear interpolation. 1costingfe produces single point estimates. For a costing brain, uncertainty quantification is essential.

6. **Wire up the SysML integration** — If fusion-tea is going to use 1costingfe, it needs either a TEAx module wrapper or a `generate_costs.py` adaptation that calls `run_costing()`. The current `adapter.py` interface shape is fine; the glue code doesn't exist yet.

---

## Bottom Line

1costingfe is a well-architected, JAX-native costing engine with genuine technical improvements over PyFECONS (differentiability, inverse power balance, multi-fuel design). But it's currently a **prototype that computes plausible numbers** rather than a **production costing brain that validates its inputs and communicates uncertainty**. The gap to close isn't in the math — it's in the engineering discipline that PyFECONS already has (validation, warnings, LSA, uncertainty) and in resolving the architectural boundary question with fusion-tea's SysML layer.
