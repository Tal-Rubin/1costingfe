# CAS220109: Direct Energy Converter (Venetian Blind) — Mirror Machine Add-On

**Date:** 2026-04-01
**Status:** Justified — subsystem build-up from Hoffman 1977 scaling laws, cross-referenced with Moir/Barr 1973–74 and MARS 1984

## Overview

Account C220109 covers the direct energy converter (DEC) for magnetic
mirror (and FRC) concepts where charged-particle exhaust escapes along
open field lines and can be electrostatically decelerated to recover
kinetic energy as electricity, bypassing the thermal cycle.

The venetian blind DEC uses angled metal ribbon grids at successively
higher retarding potentials.  Ions penetrate until they lack energy to
reach the next stage, reverse trajectory, and are collected on
high-potential electrodes.  This passively sorts ions by energy.

**Applicability:**
- Tokamak / stellarator: **\$0** (isotropic loss, no directed exhaust)
- Mirror / FRC (DT): **\$0** (not economic — see "Efficiency argument" below)
- Mirror / FRC (DD): **\$0** (marginal benefit over pure thermal)
- Mirror / FRC (DHe3): **Active** — majority charged energy fraction
- Mirror / FRC (pB11): **Active** — nearly all energy is charged

For DT and DD mirrors, the DEC adds enormous complexity for negligible
net efficiency gain (0–5 percentage points over pure thermal at
demonstrated DEC efficiencies).  See "Analysis Venetian Blind Direct
Energy Converter.md" for the full argument.

---

## Demonstrated Performance

| Configuration | Efficiency | Source |
|---|---|---|
| 22-stage collector (lab, e-beam) | 86.5% | Moir UCRL-79015 |
| 2-stage venetian blind (lab) | 65% | Barr et al. 1974 |
| 1-stage (TMX plasma test) | 48% | Moir 1984 |
| Theoretical max (venetian blind) | ~70% | Moir & Barr 1973 |

**Design point for costing:** 50% single-stage, 63% three-stage.
Single-stage is more cost-effective per Hoffman 1977 analysis.

---

## The Efficiency Argument: Why DEC is \$0 for DT/DD Mirrors

In D-T fusion, ~80% of energy leaves as 14.1 MeV neutrons (invisible
to electrostatic DEC).  The venetian blind only touches the ~20%
charged fraction:

| Cycle | Net plant efficiency |
|---|---|
| Pure thermal (blanket + steam) | ~40–45% of P_fusion |
| Hybrid (thermal for neutrons + DEC at 50% for charged) | (0.80 × 0.40) + (0.20 × 0.50) = 42% |
| Hybrid (thermal + DEC at 65%) | (0.80 × 0.40) + (0.20 × 0.65) = 45% |

The DEC adds at most ~5 percentage points for D-T at the cost of a
complex, radiation-degraded subsystem with no maintenance precedent.
Supercritical CO2 Brayton cycles (TRL 5–6, ~50% efficiency) can
capture the same gain thermally with commercial hardware.

For D-He3 (~60% charged) and p-B11 (~99% charged), the DEC becomes
the **primary** power conversion pathway, not a supplement.

---

## Historical Cost Data

### Primary Source: Hoffman 1977 (UCID-17560)

Most detailed DEC cost study available.  Provides subsystem-level
scaling equations for both single-stage (SSDC) and 3-stage venetian
blind (VBDC) converters.  All costs 1975 dollars, FOAK.

**3-Stage VBDC handling 1112 MW mirror leakage (710 MWe output):**

| Subsystem | % of Total | Cost (1975\$) | Cost (2024\$) |
|---|---|---|---|
| Vacuum tank | 37.1% | \$135M | \$783M |
| Cryo vacuum system | 37.5% | \$137M | \$795M |
| DC converter modules (grids) | 2.3% | \$8M | \$46M |
| DC power conditioning | 7.2% | \$26M | \$151M |
| Thermal panels | 8.2% | \$30M | \$174M |
| Bottoming plant | 7.7% | \$28M | \$162M |
| **Total** | **100%** | **\$364M** | **\$2,111M** |

Specific capital: \$418/kWe (1975\$) → \$2,424/kWe (2024\$)

**Optimized SSDC (thin membrane tank, 10% CX tolerance):**
Specific capital: \$135–146/kWe (1975\$) → \$783–847/kWe (2024\$)

**CPI escalation:** 1975 CPI = 53.8, 2024 CPI = 312.3 → ratio 5.80×

### Key cost equations (Hoffman Table 3):
- Vacuum tank: \$13.2/kg × tank mass (stainless, installed)
- Cryopanels: \$6,300/m² frontal area
- DC modules: \$4,000/m² frontal area
- Power conditioning: \$46/kWe
- Thermal panels: \$3,300/m²
- Bottoming plant: \$175/kWe

### Secondary: Moir & Barr 1974

Total DEC cost for 1000 MW handled: ~\$110M (1974\$), or \$110/kW of
power into the converter.  Consistent with Hoffman range.

### MARS Study (1984)

MARS tandem mirror (2600 MW fusion, 1200 MWe net) used a gridless
DEC variant.  DEC recovered 290 MWe.  Total plant: \$2.9B (1983\$) =
~\$2,400/kWe.  Barr & Moir (1980) described DEC as "a rather small
addition" to the tandem mirror since expander tanks and vacuum already
exist for plasma confinement.

---

## Integrated Mirror Machine: Shared vs. DEC-Specific Costs

**Critical insight:** In a mirror machine, the magnetic expander and
end tanks already exist for plasma confinement.  The DEC "add-on"
cost is NOT the full standalone system from Hoffman — it is only the
incremental hardware beyond what the mirror already requires.

### Shared with base mirror machine (NOT charged to C220109):
- Expander tank structure (exists for plasma exhaust handling)
- Expander/mirror coils (part of magnet system, C220103)
- Base vacuum system (required for plasma operation)

### DEC-specific add-on (charged to C220109):

| Subsystem | Scaling basis | Build-up (2024\$, 1 GWe ref) |
|---|---|---|
| Grid/collector modules | Collector area | \$10–15M |
| DC-AC power conditioning | DEC electric output | \$35–50M |
| Heat collection system | Thermal load on grids | \$8–12M |
| Incremental vacuum (cryo) | DEC volume/gas load | \$10–20M |
| Incremental tank volume | Expansion ratio | \$15–25M |
| Neutron trap shielding | DT only | \$0–5M |
| DEC gate valve | Fixed | \$0.5M |
| **Total DEC add-on** | | **\$79–128M** |

### Build-up methodology

**Grid/collector modules (\$10–15M):**
Hoffman: \$4,000/m² (1975\$) = \$23,200/m² (2024\$).  For 1 GWe mirror
with ~300 MW charged particle throughput, expansion ratio ~100, mirror
exit area ~3 m², collector area ~300 m².  Cost = 300 × \$23.2k = \$7M.
Add 50% for support structure, alignment, and electrical connections:
\$10M.  Three-stage system: ×1.5 = \$15M.

**DC-AC power conditioning (\$35–50M):**
Hoffman: \$46/kWe (1975\$) = \$267/kWe (2024\$).  Modern high-power
inverters (solar/wind industry) have driven costs down substantially.
NREL utility-scale inverter benchmark (2024): \$30–50/kWe for >100 MW
systems.  Fusion DEC operates at higher voltage (100–250 kV) requiring
specialized conversion, so use \$80–120/kWe.
At 300–500 MWe DEC output: \$24–60M, central estimate \$40M.

**Heat collection system (\$8–12M):**
Helium-cooled thermal panels on grid supports.  Similar technology to
divertor cooling but lower heat flux (grids are geometrically
transparent, so only ~10–20% of particle power deposits as heat).
Scale from divertor cost (\$60M/GWth) at ~1/6 the thermal load:
~\$10M.

**Incremental vacuum (\$10–20M):**
Cryo pumping for the larger DEC volume and charge-exchange gas load.
Hoffman: \$6,300/m² (1975\$) = \$36,500/m² (2024\$).  But modern
cryopumps are commoditized (Edwards, Leybold).  Use \$15,000/m² for
industrial cryopanels at fusion scale.  At ~1,000 m² pump area: \$15M.

**Incremental tank volume (\$15–25M):**
The DEC needs a larger end tank than basic plasma exhaust handling.
Incremental steel: ~2,000–4,000 tonnes at \$5–8/kg fabricated and
installed.  Cost: \$10–32M, central \$20M.

---

## Recommended Cost Model

### Scaling formula

    C220109 = dec_base * (p_dee / P_DEE_REF) ^ 0.7

where:
- `dec_base` = 100.0 M$ (total DEC add-on cost at reference output)
- `P_DEE_REF` = 400 MWe (reference DEC electric output)
- `p_dee` = DEC electric output in MW (from physics layer: `f_dec * eta_de * p_transport`)

C220109 is gated on `f_dec > 0`. No fuel gating is applied — the model
computes DEC cost for any fuel if the user sets `f_dec > 0`. Economic
viability is a user judgment (the DEC is not cost-effective for DT/DD
at demonstrated efficiencies, but the model does not prevent the user
from exploring this).

The 0.7 exponent reflects:
- Grid modules scale ~linearly with area (proportional to power)
- Power conditioning scales ~linearly with output
- Tank and vacuum scale sub-linearly (surface-to-volume)
- Consistent with other vendor-purchased power systems in CAS22

DEC type (venetian blind, TWDEC, ICC) is not distinguished in the cost
model. The add-on cost ranges overlap ($73-140M at ~400 MWe) because
the dominant costs (vacuum, cryo, power conditioning) are shared
infrastructure. Efficiency differences between DEC types flow through
`eta_de` in the physics layer.

### DEC grid replacement (CAS72)

DEC grids degrade under charged particle bombardment (sputtering,
helium blistering) and neutron damage. Grid replacement is modeled
as a separate CAS72 term, independent of blanket/divertor replacement:

    annual_replace_dec = dec_grid_cost * (p_dee / P_DEE_REF) ^ 0.7 / dec_grid_lifetime

**Grid lifetime (FPY) — HIGH UNCERTAINTY:** No reactor-scale data
exists for any DEC grid type. These are conservative estimates.
Primary degradation is from charged particle exhaust (sputtering,
He blistering), with neutron damage additive for DT/DD.
Sensitivity range: 0.5x to 3x.

| Fuel | dec_grid_lifetime (FPY) | Primary degradation |
|---|---|---|
| DT | 2.0 | Sputtering + 14.1 MeV neutron damage |
| DD | 3.0 | Sputtering + 2.45 MeV neutron damage |
| DHe3 | 4.0 | 14.7 MeV proton sputtering + He blistering |
| pB11 | 3.0 | 2.9 MeV alpha sputtering + severe He blistering (3 alpha per event) |

### FOAK multiplier

Apply standard contingency (10%) for FOAK per CAS29 methodology.
No additional FOAK markup — DEC components are extensions of existing
industrial technology (vacuum vessels, cryopumps, power electronics),
not novel nuclear-grade systems.

---

## Comparison with pyFECONs Values

pyFECONs (cas220109_direct_energy_converter.py) lists subsystem costs
from Post 1970 totaling \$447M with a power × flux scaling formula.
However:

1. **C220109 is currently set to \$0** with a TODO noting the
   discrepancy.  The \$447M total is never used.
2. The \$447M appears to be a **standalone DEC system** (including
   full expander tank and coils), not a mirror add-on.  For an
   integrated mirror, shared infrastructure should not be double-counted.
3. The Post 1970 values predate the more detailed Hoffman 1977 analysis
   and lack clear year-dollar basis.
4. The scaling formula `cost × system_power × (1/√flux_limit)³` has no
   documented derivation and produces unreasonable values at typical
   parameters.

**Recommendation:** Replace pyFECONs C220109 with the subsystem
build-up model above.

---

## Sensitivity and Uncertainty

| Parameter | Range | Impact on C220109 |
|---|---|---|
| DEC efficiency (1-stage vs 3-stage) | 48–65% | Higher η → more MWe recovered → larger power conditioning cost, but better economics |
| Expansion ratio | 25–400 | Higher ER → larger tank/vacuum cost, but lower heat flux on grids |
| Charge-exchange loss fraction | 1–10% | Higher tolerance → smaller cryo system → 27–46% cost reduction (Hoffman) |
| Number of stages | 1–3 | 3-stage is ~1.5× cost of 1-stage for ~30% more efficiency |
| Grid material (Cu vs Mo vs W) | — | Affects sputtering lifetime; W is most durable but hardest to fabricate |

**Overall uncertainty:** ±40% on base costs.  This is appropriate
for TRL 4–5 technology that has never been built at reactor scale.
The uncertainty is comparable to other fusion-specific components
(blanket, divertor) at similar maturity.

---

## Survivability Caveat

The cost estimate above is for initial capital.  The analysis document
("Analysis Venetian Blind Direct Energy Converter.md") identifies
seven failure modes that degrade DEC performance over time:

1. Sputtering erosion of grid geometry
2. Helium blistering compromises voltage holding
3. Secondary electron emission reduces efficiency
4. Charge-exchange neutrals bypass converter
5. Neutron damage (cumulative, for DT/DD)
6. Heat removal from thin ribbons
7. Space-charge / Debye length limits (Rosenbluth objection)

For DHe3 and pB11, failure modes 1, 4, 5 are greatly reduced
(lower neutron flux, lower sputtering from lighter ions), making
the DEC more viable as a long-lived component.

**Replacement cycle:** Modeled as a separate CAS72 term using the
`dec_grid_lifetime` table in the Recommended Cost Model section above.

---

## References

- Hoffman, M.A., "Electrostatic Direct Energy Converter Performance
  and Cost Scaling Laws," UCID-17560, Lawrence Livermore Laboratory,
  August 1977.  [OSTI 7218298](https://www.osti.gov/biblio/7218298)
- Moir, R.W. & Barr, W.L., "Venetian-blind direct energy converter
  for fusion reactors," Nuclear Fusion 13, 35–45, 1973.
  [OSTI 4563116](https://www.osti.gov/biblio/4563116)
- Barr, W.L. et al., "A Preliminary Engineering Design of a Venetian
  Blind Direct Energy Converter for Fusion Reactors," IEEE Trans.
  Plasma Sci. PS-2(2), 71–92, 1974.
  [OSTI 4232519](https://www.osti.gov/biblio/4232519)
- Moir, R.W., "Direct Conversion of Fusion Energy" (review),
  UCRL-79015, Lawrence Livermore Laboratory.
  [OSTI 7341986](https://www.osti.gov/biblio/7341986)
- Barr, W.L. & Moir, R.W., "Technology of direct conversion for
  mirror reactor end-loss plasma," 1980.
  [OSTI 6894579](https://www.osti.gov/biblio/6894579)
- Logan, B.G. et al., "Mirror Advanced Reactor Study (MARS),"
  UCRL-53480, July 1984.
  [OSTI 5981974](https://www.osti.gov/biblio/5981974)
- Post, R.F., "Mirror systems: fuel cycles, loss reduction and energy
  recovery," in Nuclear Fusion Reactors, pp. 99–111, Thomas Telford
  Publishing, 1970.
- Woodruff, S., "A Costing Framework for Fusion Power Plants,"
  arXiv:2601.21724, January 2026.
- Realta Fusion, "Axisymmetric ferromagnetic venetian blinds,"
  US Patent 12,166,398 B2, 2025.
- NREL, "Utility-Scale Solar Photovoltaics — 2024 Cost Benchmark."
- Hoffman, M.A. & Hamilton, G.W., "Direct Energy Conversion Cost
  Estimates," J. Energy 2(5), 293, 1978.
