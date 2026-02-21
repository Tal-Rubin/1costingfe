# CAS220112: Isotope Separation Plant — Zeroed

**Date:** 2026-02-20
**Status:** Zeroed (intentionally removed)

## Decision

CAS220112 is set to zero for all fuel cycles. There is no on-site isotope
separation plant. All isotope procurement is modeled as market purchase in
CAS80 at enriched $/kg prices.

## Rationale

CAS220112 (on-site separation plant capital) and CAS80 (isotope unit costs at
enriched market prices) double-count the cost of isotope separation. If CAS80
pays $2,175/kg for deuterium, that price already includes the supplier's
extraction and enrichment costs. Building a $300M on-site D2O plant AND paying
market price for enriched deuterium charges for separation twice.

The correct model is one or the other:
- **Market purchase**: CAS80 at enriched $/kg, CAS220112 = 0
- **On-site plant**: CAS220112 capital cost, CAS80 at raw feedstock prices

We chose market purchase because on-site separation plants are not justified
at fusion-scale consumption rates.

## Consumption Analysis (1 GWe reference, p_fus=2300 MW, avail=0.85)

| Isotope | kg/yr consumed | Global enriched supply | On-site justified? |
|---------|---------------|----------------------|-------------------|
| Deuterium | 73-194 | ~80,000+ kg/yr (D2O plants) | No — <0.3% of global |
| Li-6 | 219 | ~1-2 t/yr enrichment | No for FOAK; fleet concern |
| He-3 | 105 | ~15 kg/yr (T decay) | N/A — supply not viable |
| Protium | 74 | millions of t/yr (H2) | No — commodity |
| B-11 | 811 | research-only (kg scale) | No — centralized, not on-site |

### Key observations

**Tritium** is bred on-site in the blanket and processed by the fuel handling
system (CAS220500). This is not an isotope *purchase* — it is already modeled
in CAS220101 (breeding blanket) and CAS220500 (fuel handling + T containment).

**Deuterium** is produced at industrial scale by D2O plants (India alone has
~400 t/yr D2O capacity). A fusion plant consuming 73 kg/yr is a trivial customer.

**Li-6** enrichment capacity (~1-2 t/yr globally) could be stressed by a fleet
of DT reactors (each needing ~219 kg/yr). For fleet scenarios, a centralized
COLEX facility shared across plants would make sense — but this is a fleet-level
investment, not a per-plant CAS22 capital item.

**B-11** has no industrial-scale enrichment market today (LPP Fusion paid $600/g
for 93g lab-scale in 2018). However, B-11 enrichment is really B-10 *depletion*
— the same chemical exchange distillation used for B-10 production (proven at
~480 kg/yr) yields >99% B-11 as tails. A centralized facility serving a fleet
of pB11 plants is straightforward and would not be on-site at each plant (B-11
is inert powder, trivial to ship).

## CAS80 unit costs (retained, market prices)

| Isotope | $/kg | Basis |
|---------|------|-------|
| Deuterium | 2,175 | STARFIRE (1980) inflation-adjusted |
| Li-6 (enriched) | 1,000 | 90% enriched Li-6 |
| He-3 | 2,000,000 | Scarcity pricing ($2,000/g) |
| Protium | 5 | Commodity H2 |
| B-11 (enriched) | 10,000 | FOAK estimate ($10/g, B-10 tails) |

## Sources

- LPP Fusion (2018): 93g of 99.9% B-11 at $600/g, custom lab production
  (Russia isotopic purification + Czech Republic decaborane synthesis)
- OSTI technical reports: B-10 chemical exchange distillation at 40 kg/month
- India Heavy Water Board: ~400 t/yr D2O production capacity
- pyfecons CAS220112: $300M D extraction, $100M Li-6, $125M B-11 (not used)

## What was removed

From `defaults.py` and `costing_constants.yaml`:
- `deuterium_extraction_base` (was $15M)
- `li6_enrichment_base` (was $25M)
- `he3_extraction_base` (was $0M)
- `protium_purification_base` (was $5M)
- `b11_enrichment_base` (was $20M)

The CAS220112 key is still emitted in the result dict (as 0.0) for compatibility.
