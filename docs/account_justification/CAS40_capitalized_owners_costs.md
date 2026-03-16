# CAS40: Capitalized Owner's Costs (COC)

**Date:** 2026-03-16
**Status:** Implemented

## Account Structure

CAS40 covers costs incurred by the plant owner during the construction period to prepare for plant operation. Sub-accounts per GEN-IV EMWG (2007) Table 1.2:

| CAS | Description |
|-----|-------------|
| 41 | Staff Recruitment and Training |
| 42 | Staff Housing |
| 43 | Staff Salary-Related Costs |
| 44 | Other Owner's Capitalized Costs |
| 49 | Contingency on Owner's Costs |

Scope of each sub-account (from EMWG Chapter 6, p.151, and INL Sort_67398, p.77):

- **C41 (Staff Recruitment and Training):** Costs to recruit and train plant operators before plant startup or commissioning (Account 33) or demonstration tests (Account 34). This is the single largest sub-account.
- **C42 (Staff Housing):** Relocation costs, camps, or permanent housing provided to permanent plant operations and maintenance staff.
- **C43 (Staff Salary-Related Costs):** Taxes, insurance, fringes, benefits, and any other salary-related costs during the pre-operational hiring period.
- **C44 (Other Owner's Costs):** Catch-all for additional owner's capitalized costs not captured elsewhere.
- **C49 (Contingency):** Assessment of additional costs to achieve the desired confidence level.

**Key boundary note:** The EMWG explicitly states that "the utility's (owner's) pre-commissioning costs are covered elsewhere in the TCIC sum as a capitalized owner's cost (Account 40)" — distinguishing owner's pre-commissioning costs from A/E and vendor commissioning costs in Account 33.

**CAS40 vs CAS70 boundary (double-counting avoidance):**

CAS40 and CAS70 share the same underlying staff — the people recruited and trained under CAS40 become the operating staff whose annual costs are CAS70. The boundary is the commercial operation date (COD):

| Account | Time Period | Cost Type | Example |
|---------|-------------|-----------|---------|
| CAS41 | Pre-COD | One-time capital | Recruiting fees, training program costs |
| CAS42 | Pre-COD | One-time capital | Relocation/housing for new hires |
| CAS43 | Pre-COD | One-time capital | Salary + benefits during ~1.5yr training period |
| CAS71-73 | Post-COD | Annual recurring | Salary + benefits during plant operation |

There is no double-counting: CAS40 is the cost of *building* the operating organization; CAS70 is the cost of *running* it. Both are derived from the same staffing analysis (see `CAS70_staffing_and_om_costs.md`) but cover different time periods. The INL Sort_67398 report uses exactly this approach — the same staff headcount drives both accounts.

## Source Documents

### Primary sources (read directly)

1. **Woodruff, S.** "A Costing Framework for Fusion Power Plants," arXiv:2601.21724v2, January 2026.
   - Table 2: CAS40 = $185M (8% of TCC) for a 637 MWe plasma-jet MIF example.
   - MARS comparison: CAS40 = $610M (6.7% of TCC) for MARS.
   - Uses pyfecons LSA-fraction methodology.

2. **pyfecons source code.** `cas40_capitalized_owner.py` and `lsa_levels.py`.
   - `C400000LSA = fac_91[LSA-1] * C200000`
   - fac_91 values: LSA1=0.113, LSA2=0.120, LSA3=0.128, LSA4=0.151
   - Sub-accounts C41-C44 all set to zero; the LSA fraction IS the total.
   - Code contains `# TODO determine cost basis, ask simon` and `# TODO explanation for this section?`

3. **GEN-IV EMWG (2007).** "Cost Estimating Guidelines for Generation IV Nuclear Energy Systems," GIF/EMWG/2007/004, Rev 4.2. [PDF](https://www.gen-4.org/gif/upload/docs/application/pdf/2013-09/emwg_guidelines.pdf)
   - Account 40 defined on p.31, detailed on p.151.
   - No specific percentage guidance for top-down estimation of Account 40 (unlike CAS30 and CAS70, which have more prescriptive formulas).
   - Overnight cost = Accounts 10 + 20 + 30 + 40 + 50 (p.29).

4. **Shropshire, D. et al.** "Advanced Nuclear Reactor Cost Estimation: Sodium-Cooled Fast Reactor Case Study," INL, Sort_67398, 2024. [PDF](https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_67398.pdf)
   - CAS40 sub-accounts estimated bottom-up for a 165 MWe (single) / 1,243 MWe (multi-pack) SFR.
   - C41 = $136/kWe (1.5 yrs salary × 110% staff, plus 25% recruiting fee)
   - C42 = $10/kWe ($40k relocation × 50% of staff)
   - C43 = $72/kWe (57-59% of salary for benefits)
   - C49 = $44/kWe (20% contingency)
   - **Total CAS40 = $262/kWe** (vs. EEDB PWR12-BE = $357/kWe)
   - States: "capitalized owner's costs are about 5-9% of TOC" (p.77)
   - At BOAK total of $4,537/kWe (Accounts 10-60), CAS40 is **5.8% of TCIC**.
   - **493 staff at 1,243 MWe** drives the CAS40 total — staffing is the dominant cost driver.

5. **CAS71-73 staffing analysis.** `docs/account_justification/CAS70_staffing_and_om_costs.md`.
   - Fuel-specific staffing estimates at 1 GWe reference, derived from S-PRISM fission baseline scaled by regulatory burden and fusion-specific complexity.
   - The same staff that CAS40 recruits and trains are the staff that CAS70 pays annually.

6. **pB11 regulatory risk analysis.** `docs/analysis/pb11_side_reactions_nrc_regulatory_risk.md`.
   - All fusion fuels under Part 30 (not Part 50) per Feb 2026 proposed rule.
   - pB11 has ~10,000x fewer neutrons than DT — minimal regulatory staffing burden.
   - Regulatory burden gradient: pB11 (near-industrial) < DHe3 < DD < DT (heaviest Part 30).

### Secondary sources (cited in literature)

7. **Miller, R.L.** "Economics and Costing," Chapter 3, STARLITE Final Report, UCSD, 1997. — Origin of the LSA factor table (fac_91 through fac_98). Values verified against ARIES-RS output.
8. **Piet, S.J.** "Inherent/Passive Safety for Fusion," Fusion Technology 10(3P2B), November 1986. — Defined the 4 LSA levels.
9. **Waganer, L.M.** "ARIES Cost Account Documentation," UCSD-CER-13-01, June 2013. [PDF](https://qedfusion.org/LIB/REPORT/ARIES-ACT/UCSD-CER-13-01.pdf)
10. **IAEA.** "Staffing of Nuclear Power Plants and the Recruitment, Training and Authorization of Operating Personnel," 1991. — Basis for 110% overhire factor and staggered recruitment approach.

## Existing Methods

### Method A: LSA fraction of direct cost (Miller/ARIES/pyfecons)

pyfecons computes CAS40 as a fraction of CAS20 (total direct cost), using the same LSA factor table used for CAS30 sub-accounts:

```
C40 = fac_91[LSA] * CAS20
```

| LSA | fac_91 | Interpretation |
|-----|--------|---------------|
| 1 | 0.113 | Inherently safe, lowest regulatory burden |
| 2 | 0.120 | Inherently safe with some active safety |
| 3 | 0.128 | Active safety, moderate regulatory burden |
| 4 | 0.151 | Fission-like, full nuclear-grade |

**Critical flaw:** fac_91 is used for **both** CAS32 (Construction Supervision) **and** CAS40 (Owner's Costs) in pyfecons. These are fundamentally different scopes. Construction supervision is contractor-provided field management; owner's costs are pre-operational staffing, training, and housing. The shared factor is an artifact of mapping the old ARIES "Account 91" (which was a combined indirect/owner bucket) into the GEN-IV COA. The pyfecons code even flags this: `# TODO determine cost basis, ask simon`.

**Why this produces wrong results for fusion:** The fac_91 values were calibrated against fission plants with ~500 staff. Applying 11-15% of TDC to a fusion plant implicitly assumes the same staffing ratio — but our CAS71-73 analysis shows fusion plants have 1/4 to 1/8 the staff of fission. A DT fusion plant with 117 staff should not have the same owner's cost fraction as a fission plant with 493 staff.

### Method B: Bottom-up staffing (INL Sort_67398)

The INL report estimates CAS40 from first principles for a 1,243 MWe SFR (493 staff):

| Sub-account | Basis | $/kWe |
|-------------|-------|-------|
| C41 | 1.5 yr salary × 110% staff + 25% recruiting | 136 |
| C42 | $40k relocation × 50% of staff | 10 |
| C43 | 57-59% of salary (benefits, taxes, insurance) | 72 |
| C44 | None | 0 |
| C49 | 20% contingency | 44 |
| **Total** | | **262** |

**Methodology:** The pre-operational period is ~1.5 years before COD. IAEA recommends hiring 110% of needed staff (10% attrition buffer during training). Recruitment follows a staggered approach: professionals first, then operators, then maintenance. Total pre-operational cost per employee works out to roughly **3.2× annual total compensation** (salary + benefits).

Verification: INL CAS40 (pre-contingency) = $218/kWe × 1,243 MWe = $271M. Annual total comp ≈ 493 × $170k = $83.8M. Ratio: $271M / $83.8M ≈ 3.2×.

## Derivation: Staffing-Based CAS40 for Fusion by Fuel Type

### Step 1: Staffing inputs from CAS71-73 analysis

From `CAS70_staffing_and_om_costs.md`, at 1 GWe reference:

| Fuel | Staff | Annual Labor+Benefits | Annual Salary | Annual Benefits |
|------|-------|-----------------------|---------------|-----------------|
| pB11 | 59 | $10.1M | $6.5M | $3.6M |
| DHe3 | 69 | $11.7M | $7.5M | $4.2M |
| DD | 94 | $15.9M | $10.2M | $5.7M |
| DT | 117 | $19.7M | $12.6M | $7.1M |

Salary = Labor+Benefits − (staff × $61k benefits). Benefits = staff × $61k/yr (BLS March 2023, per INL/RPT-23-74316).

### Step 2: Apply INL methodology to fusion staffing

**C41 (Recruitment + Training):**
- Pre-operational training period: 1.5 years (INL/IAEA basis)
- Overhire factor: 1.10 (IAEA recommendation for training attrition)
- Training salary cost: 1.5 yr × 1.10 × annual_salary
- Recruitment fee: 25% of annual salary × 1.10 (nationwide recruiting, per INL)
- **C41 = 1.10 × (1.5 + 0.25) × annual_salary = 1.925 × annual_salary**

**C42 (Staff Housing/Relocation):**
- Relocation cost: $40k/person (INL basis, 2023$)
- Relocation fraction: 50% of staff (fusion-specific systems require specialist hiring from outside local area)
- Overhire factor: 1.10
- **C42 = $0.04M × 0.50 × 1.10 × staff_count = $0.022M × staff_count**

**C43 (Salary-Related Costs during pre-op):**
- Benefits during training: 1.5 yr × 1.10 × annual_benefits
- **C43 = 1.65 × annual_benefits**

| Fuel | C41 | C42 | C43 | **CAS40 Total** | **$/kWe** |
|------|-----|-----|-----|----------------|-----------|
| pB11 | $12.5M | $1.3M | $5.9M | **$19.7M** | **$20** |
| DHe3 | $14.4M | $1.5M | $6.9M | **$22.9M** | **$23** |
| DD | $19.6M | $2.1M | $9.4M | **$31.1M** | **$31** |
| DT | $24.3M | $2.6M | $11.8M | **$38.6M** | **$39** |

### Step 3: Validate against INL fission reference

| Plant | Staff | CAS40 (pre-contingency) | $/kWe | % of TDC |
|-------|-------|------------------------|-------|----------|
| INL SFR (1,243 MWe) | 493 | $271M | $218 | 10.9% |
| DT fusion (1 GWe) | 117 | $39M | $39 | ~1% |
| pB11 fusion (1 GWe) | 59 | $20M | $20 | ~0.5% |

The fusion values are much lower per-kWe than fission. This is correct — the whole thesis of the CAS71-73 analysis is that fusion plants (especially low-neutron fuels) have dramatically less staffing than fission:
- DT fusion: ~1/4 of fission staffing → ~1/6 of fission CAS40/kWe (also lower avg salary due to staff mix)
- pB11 fusion: ~1/8 of fission staffing → ~1/11 of fission CAS40/kWe

The fission CAS40 is dominated by the cost of training ~200+ radiation protection, security, regulatory compliance, and nuclear QA staff that fusion plants under Part 30 simply do not need.

### Comparison of all methods

For a 1 GWe DT fusion plant with ~$4B direct cost:

| Method | CAS40 | $/kWe | % of TDC | Source |
|--------|-------|-------|----------|--------|
| pyfecons fac_91 LSA=1 | $452M | $452 | 11.3% | ARIES/Miller |
| pyfecons fac_91 LSA=2 | $480M | $480 | 12.0% | ARIES/Miller |
| INL SFR (fission, scaled to 1 GWe) | ~$218M | $218 | 5.5% | Sort_67398 |
| **Staffing-based (this analysis)** | **$39M** | **$39** | **~1%** | CAS71-73 + INL methodology |

**Why the 10x difference from fission-calibrated methods?**

The pyfecons/ARIES approach applies a fission-calibrated 11-15% factor that implicitly assumes ~500 staff per GWe. The INL bottom-up for fission explicitly shows 493 staff driving $218/kWe. Our CAS71-73 analysis shows fusion needs only 59-117 staff — the staffing-based approach correctly propagates this reduction into CAS40.

The fraction-of-TDC methods are not "wrong for fission" — they give reasonable results for 500-staff nuclear plants. They are wrong for fusion because they fail to account for the dramatic staffing reduction that comes from Part 30 regulation and the elimination of fission-specific safety systems.

## Assessment

**What we know with confidence:**

1. CAS40 is dominated by pre-operational staffing costs (C41 + C43 > 95% of total in INL bottom-up).
2. Pre-operational staffing scales with operations staffing — the people you recruit and train under CAS40 are the people you pay under CAS70.
3. We have defensible fuel-specific staffing estimates from the CAS71-73 analysis, grounded in the S-PRISM fission reference, CCGT conventional reference, and NRC Part 30 regulatory framework.
4. The INL methodology (1.5yr pre-op, 10% overhire, 25% recruiting, $40k relocation, 58% benefits) provides a well-documented cost model that we can apply to our fusion staffing numbers.
5. Fusion plants have dramatically less staffing than fission (59-117 vs 493) — CAS40 should reflect this.

**What is NOT defensible:**

1. Applying fission-calibrated TDC fractions (11-15%) to fusion plants. This implicitly assumes fission-level staffing.
2. Using fac_91 for CAS40. This was a pyfecons accounting artifact (same factor for CAS32 and CAS40), flagged with TODO comments in the pyfecons source.
3. Sub-account precision beyond C41/C42/C43 totals — the individual line items are calibrated from fission and adapted to fusion; they are estimates, not measurements.

**Fuel-type differentiation is driven by neutron-related staffing:**

The CAS71-73 analysis shows that the staffing gradient across fuels is almost entirely driven by neutron production and its consequences (radiation protection, tritium handling, radwaste, remote maintenance). This same gradient naturally propagates into CAS40:

| Fuel | Key staffing drivers | Staff | CAS40 |
|------|---------------------|-------|-------|
| pB11 | Near-industrial; RSO only; no tritium; minimal activation | 59 | $20M |
| DHe3 | Light HP program; minor tritium from DD side reactions; some activation | 69 | $23M |
| DD | Full HP program; tritium handling; activated components; some remote handling | 94 | $31M |
| DT | Tritium breeding+processing; full rad protection; remote handling; largest HP dept | 117 | $39M |

The DT/pB11 ratio of ~2:1 in CAS40 exactly mirrors the ~2:1 ratio in staffing (117/59), which in turn is driven by the ~10,000:1 ratio in neutron production. This internal consistency across CAS40 and CAS70 — both anchored to the same staffing analysis — is a strength of the approach.

## Recommendation for 1costingfe

Implement **fuel-specific owner's cost coefficients** derived from the CAS71-73 staffing analysis and INL CAS40 methodology:

```python
def cas40_owner(cc, fuel, p_net):
    """CAS40: Capitalized owner's costs. Returns M$.

    Pre-operational costs to recruit, train, house, and compensate
    the plant operations staff before COD. Derived from the CAS71-73
    staffing analysis applied through the INL CAS40 methodology
    (1.5yr pre-op, 10% overhire, 25% recruiting, 58% benefits).

    Uses the SAME staffing basis as CAS70 annual O&M — CAS40 covers
    pre-COD costs, CAS70 covers post-COD costs. No double-counting.

    See docs/account_justification/CAS40_capitalized_owners_costs.md
    """
    return cc.owner_cost(fuel) * (p_net / 1000.0) ** 0.5
```

New `CostingConstants` fields (M$ at 1 GWe reference, 2023$):

| Parameter | Value | Derivation |
|-----------|-------|------------|
| `owner_cost_dt` | 39.0 | 117 staff, $12.6M salary, $7.1M benefits → C41+C42+C43 |
| `owner_cost_dd` | 31.0 | 94 staff, $10.2M salary, $5.7M benefits → C41+C42+C43 |
| `owner_cost_dhe3` | 23.0 | 69 staff, $7.5M salary, $4.2M benefits → C41+C42+C43 |
| `owner_cost_pb11` | 20.0 | 59 staff, $6.5M salary, $3.6M benefits → C41+C42+C43 |

Plus `owner_cost(fuel)` accessor method on `CostingConstants`, following the existing pattern for `om_cost(fuel)`, `licensing_cost(fuel)`, etc.

**Power scaling:** Power-law with exponent 0.5:

```
CAS40 = owner_cost(fuel) * (P_net / 1 GWe)^0.5
```

Staffing does not scale linearly with plant size — it exhibits significant economy of scale. The INL SFR data (Sort_67398, Table in Section 2) shows:

| Plant size | Staff | Staff/GWe |
|-----------|-------|-----------|
| 165 MWe (1 rx) | 236 | 1,430 |
| 1,243 MWe (4 rx) | 493 | 397 |
| 3,108 MWe (10 rx) | 1,040 | 335 |

Administration, technical, and offsite staff have a large fixed component (~50% fixed + 50% proportional to reactor count, per INL). Fitting a power law to the endpoints (165 → 3,108 MWe) gives α ≈ 0.5. This same exponent applies to both CAS40 (pre-operational staffing costs) and CAS70 (annual O&M costs), since both are driven by total staff count.

Example CAS40 values for DT at different plant sizes:

| Plant size | CAS40 | $/kWe |
|-----------|-------|-------|
| 200 MWe | $17.4M | $87 |
| 500 MWe | $27.6M | $55 |
| 1,000 MWe | $39.0M | $39 |
| 2,000 MWe | $55.2M | $28 |

**Why this approach:**

1. **Internally consistent** — uses the same staffing basis as CAS70, avoiding double-counting by construction.
2. **Fuel-dependent** — captures the dramatic staffing difference between DT (full neutron+tritium burden) and pB11 (near-industrial), driven by the regulatory and safety analysis in `pb11_side_reactions_nrc_regulatory_risk.md`.
3. **Auditable** — every number traces back to the CAS71-73 staffing analysis and INL methodology, not opaque Miller/ARIES factors.
4. **Appropriately modest** — these are pre-conceptual estimates for plants that don't exist yet. The staffing-based approach is transparent about what it assumes.

**Reference values:**

For a 1 GWe DT tokamak: CAS40 = $39M ($39/kWe). This is far below fission ($218/kWe for the INL SFR) because fusion under Part 30 needs ~117 staff vs ~493 for fission — the elimination of armed security, licensed operators, and full nuclear-grade QA infrastructure accounts for the difference.

For a 1 GWe pB11 plant: CAS40 = $20M ($20/kWe) — reflecting a near-industrial plant with 59 staff and RSO-only radiation protection.
