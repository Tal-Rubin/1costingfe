# CAS21 Building & Civil Works Cost Research

**Date:** 2026-03-20
**Purpose:** From-scratch research on building/civil works costs across construction grades, to support rebuilding the CAS21 cost model for fusion power plants.

---

## 1. Gas Turbine Combined Cycle (CCGT) Civil Works Costs

### Total CCGT Capital Costs
- **EIA 2015 average:** $614/kW for combined-cycle units (2015$)
- **EIA 2025 projection:** $1,463/kW
- **Gas Turbine World 2024:** $670/kW (1,680 MW) to $1,579/kW (34 MW) -- strong scale effect
- **Recent US projects:** Often exceeding $2,000/kW due to supply chain and labor inflation

### Civil Works as Fraction of Total
| Component | % of EPC | Implied $/kW (at $1,000/kW) |
|-----------|--------:|---:|
| Core turbine equipment | ~40% | ~$400 |
| Balance of plant (aux systems) | 25-35% | ~$300 |
| **Civil works & site development** | **15-25%** | **$150-250** |
| Electrical infrastructure | 12-20% | ~$160 |
| Planning, permitting, commissioning | 10-15% | ~$125 |

Source: Thunder Said Energy, Gas Turbine World, EIA

### What Civil Works Includes for CCGT
- **Site preparation:** Grading, foundations, roads, drainage. Varies dramatically by location.
- **Turbine building:** Pre-engineered steel structure, crane-served. Relatively simple.
- **Control room:** Small, conventional commercial construction.
- **HRSG foundations:** Heavy concrete pads for heat recovery steam generators.
- **Cooling tower / condenser:** Concrete basin, tower structure.
- **Switchyard:** Steel structures, gravel pad.

### Key Takeaway
CCGT civil works are **industrial-grade construction** -- pre-engineered steel buildings, conventional concrete, no special QA. At $150-250/kW, this represents the **floor** for power plant buildings cost. A CCGT "building" is essentially a weather enclosure over equipment.

---

## 2. Supercritical Coal Plant Building Costs (NETL Case B12A)

### Source
DOE/NETL "Cost and Performance Baseline for Fossil Energy Plants," Rev 4 (2019). Case B12A = supercritical pulverized coal, 3500/1100/1100 psig/F/F steam cycle, ~550 MWe net.

### NETL Code of Accounts
NETL uses a 17-account system. **Account 14 = Buildings & Structures.** This is separate from site improvements and includes:
- Boiler building
- Turbine building
- Administration/control building
- Maintenance/warehouse
- Pump houses
- Stack foundation

### Coal Plant Building Costs (from NETL baselines and literature)
- **Total plant cost (Case B12A):** ~$3,500-3,800/kW (2018$)
- **Account 14 (Buildings & Structures):** Typically 5-8% of total plant cost for coal plants
- **Implied buildings cost:** ~$175-300/kW
- **Site improvements (Account 13):** Additional ~$25-50/kW

The pyFECONS framework references NETL Case B12A for calibrating fusion building costs and arrives at ~$714/kW total for buildings -- but this includes items well beyond what NETL Account 14 covers (e.g., reactor containment-equivalent, hot cell, cryogenics building).

### Key Takeaway
Coal plant buildings are **industrial/commercial grade** construction with some heavy industrial elements (boiler house must support heavy equipment, turbine hall needs large crane). Building costs at $175-300/kW are modestly above CCGT levels due to more buildings and heavier structural requirements.

---

## 3. Nuclear Plant Building Costs (10 CFR Part 50)

### Overall Nuclear Capital Costs
| Plant | $/kW (overnight) | Year | Notes |
|-------|--:|---:|-------|
| AP1000 "should cost" (NOAK) | $4,300 | 2018$ | MIT CANES estimate, nth-of-a-kind |
| AP1000 10th unit | $2,900 | 2018$ | Learning curve benefit |
| Vogtle 3&4 (actual) | ~$9,300 | 2021$ | FOAK, massive overruns |
| EIA reference case | $7,821 | 2023$ | US reference projection |
| NREL ATB 2024 | ~$7,000 | 2024$ | Central estimate |

### Account 21 (Structures & Improvements) in Nuclear Plants
The EEDB (Energy Economic Data Base) code-of-accounts system uses **Account 21** for structures. Sub-accounts include:
- 211: Containment building (largest single structure cost)
- 212: Shield/reactor building
- 213: Turbine-generator building
- 214: Security building
- 215: Reactor service (auxiliary) building
- 216: Waste processing building
- 217: Fuel storage building

### Structures as Fraction of Nuclear Total
- **NREL ATB 2024:** Structures = 16.1% of total capital for 1 GWe fission
  - At ~$7,000/kW total --> **~$1,130/kW for structures**
- **IFP analysis:** Structures are roughly 1/3 of direct costs, which are themselves ~50-67% of total. Implies structures ~17-22% of overnight.
- **EEDB PWR12:** Account 21 represents a significant fraction of direct costs. Indirect costs (engineering, construction management) were 50-100% of direct costs depending on experience level.
- **ITER:** Buildings estimated at 15-20% of ~EUR 20B = EUR 3-4B

### Nuclear vs. CCGT Building Costs
| Metric | CCGT | Nuclear (AP1000 class) | Ratio |
|--------|-----:|-----:|------:|
| Total plant $/kW | $1,000-1,500 | $5,000-8,000 | 5-6x |
| Buildings $/kW | $150-250 | $800-1,300 | 4-6x |
| Buildings % of total | 15-25% | 15-20% | Similar |

The nuclear building cost premium is **4-6x** over CCGT, driven by nuclear-grade construction requirements (see Section 5).

### INL SFR Study (Sort_67398)
"First-Principles Cost Estimation of a Sodium Fast Reactor Nuclear Plant" by Strategic Analysis Inc. for INL. Covers a 471 MWt / 165 MWe sodium-cooled fast reactor. Uses DFMA (Design for Manufacture and Assembly) methodology for buildings and site structures. Specific cost data is in the full report (not available in search excerpts), but the study uses first-principles bottom-up costing rather than top-down scaling.

---

## 4. Industrial Facility Construction Benchmarks

### Cost per Square Foot by Facility Type (2024-2025 US$)

| Facility Type | $/sq ft | Grade | Source |
|--------------|--------:|-------|--------|
| Light industrial warehouse | $77-100 | Standard | Cushman & Wakefield 2025 |
| Distribution center | ~$257 | Standard | Industry average |
| Light industrial / commercial | ~$286 | Standard | Industry average |
| Heavy manufacturing (crane-served) | $200-400 | Industrial | RS Means, industry |
| Data center | $600-1,100 | High-spec industrial | Cushman & Wakefield 2024 |
| AI data center (liquid cooled) | $1,000-2,000+ | High-spec industrial | Industry 2025 |
| Hospital / medical facility | $400-800 | Commercial/medical | Industry average |
| Semiconductor fab (building only) | $2,000-4,000 | Cleanroom industrial | Industry estimates |
| Semiconductor fab (total w/ equipment) | $10,000-20,000 | Ultra-clean | Industry estimates |

### Data Center Cost per MW of IT Load
- **Standard data center:** $7-12M per MW of IT load (2024$)
- **AI/high-density facility:** $15-20M+ per MW

### Key Comparisons for Fusion Buildings

| Fusion Building | Closest Industrial Analog | Industrial $/sq ft | Fusion implied $/sq ft |
|----------------|--------------------------|---:|---:|
| Turbine building | Heavy manufacturing / crane hall | $200-400 | $250-350 |
| Fusion heat island | Heavy industrial + cleanroom elements | $400-800 | $500-700 |
| Hot cell | Specialized shielded facility | $1,000-3,000 | $1,500-2,500 |
| Control room | Commercial office / data center | $300-600 | $400-500 |
| Administration | Commercial office | $200-350 | $200-300 |
| Cryogenics building | Industrial / cold storage | $200-400 | $300-400 |
| Warehouse / site services | Light industrial | $100-200 | $100-150 |

---

## 5. Nuclear-Grade vs. Industrial-Grade Construction Cost Drivers

### 5.1 Seismic Category I Requirements

**What it means:** SSCs designated Seismic Category I must remain functional after the Safe Shutdown Earthquake (SSE). Design uses site-specific seismic hazard analysis, dual-level design (OBE + SSE).

**Cost impact:**
- Pure seismic upgrade cost: **~2% of total plant cost** for going from 0.2g SSE to 0.6g SSE (per OECD/NEA study)
- This is modest because seismic design is a well-understood structural engineering discipline
- The real cost is not the seismic design itself but the **QA and documentation** required to demonstrate seismic qualification

**Fusion relevance:** DT fusion plants under Part 30 (materials license, not Part 50 reactor license) would likely NOT require Seismic Category I. Industrial seismic codes (IBC/ASCE 7) would apply. This eliminates the nuclear seismic premium.

### 5.2 Nuclear QA (NQA-1 / 10 CFR 50 Appendix B)

**What it means:** Every safety-related component requires:
- Qualified suppliers with NQA-1 programs
- Full traceability of materials (heat numbers, mill certs)
- Qualified welding procedures and welder certifications
- In-process inspection holds
- Extensive documentation packages
- Independent verification and validation

**Cost impact:**
- QA costs = **23% of concrete cost** and **41% of structural steel cost** on nuclear plants (Dawson 2017, cited by Breakthrough Institute)
- NQA-1 qualified components can be **up to 50x** more expensive than commercial-grade equivalents for some items (EPRI finding, cited by IFP/Construction Physics)
- The 50x figure applies to specific mechanical/electrical components; for bulk commodities (concrete, rebar, structural steel) the premium is lower
- Nuclear-grade concrete: **$527/m^3** vs. non-nuclear $352/m^3 = **1.5x material cost premium** (2022 analysis)
- But installation time for nuclear concrete is **33-105% longer** than conventional, roughly doubling the installed cost

**Overall NQA-1 multiplier on structures:**
- Material cost: 1.3-1.5x
- Installation labor: 1.5-2.0x (longer durations + more expensive professionals)
- QA/QC overhead: adds 25-40% to installed cost
- **Total: approximately 2.0-3.5x for bulk structural work**
- **For specific safety-related components (valves, pumps, instruments): 5-50x**

**Fusion relevance:** DT fusion under Part 30 requires NQA-1 only for specific tritium-containing systems, not for the entire plant. pB11 fusion would need no NQA-1 at all.

### 5.3 Containment Structures

**What it means for fission:** A massive reinforced concrete and steel-lined pressure boundary designed to contain the release of radioactive material from a core damage accident. Typically 1.2-1.5m thick reinforced concrete walls with 6-8mm steel liner. Vogtle AP1000: 10-foot thick basemat, 4-foot thick walls.

**Cost impact:**
- Containment is typically the **single most expensive structure** in a nuclear plant
- AP1000 was designed to use ~1/5 the concrete and rebar of older PWR designs, specifically to reduce this cost
- Containment alone can represent **20-30% of Account 21** structures costs

**Fusion relevance:** Fusion does not need a fission-style containment building. Even DT fusion:
- No fission product inventory requiring pressure containment
- Tritium inventory is grams, not tons of fission products
- No risk of steam explosions or hydrogen deflagration from core damage
- Confinement (not containment) is the applicable concept -- essentially a sealed building with negative pressure and filtered exhaust

### 5.4 Safety-Related Concrete Specifications

**Nuclear concrete vs. conventional:**
| Parameter | Nuclear (ACI 349) | Conventional (ACI 318) |
|-----------|-------------------|----------------------|
| Mix design | Tightly controlled water/cement ratio | Standard |
| Rebar density | Very dense (hard to pour around) | Standard |
| Rebar inspection | Every splice inspected, documented | Sampling-based |
| Pour documentation | Full batch traceability | Standard QC |
| Curing requirements | Strictly controlled | Standard |
| Testing | Extensive, every pour | Sampling-based |
| Cost per cubic yard | ~$530/m^3 ($405/yd^3) | ~$350/m^3 ($270/yd^3) |

Installation time premium: **33-105%** longer for nuclear concrete vs. conventional.

**Fusion relevance:** Only the hot cell and tritium-containing areas of a DT plant would need anything approaching nuclear concrete specs. Most buildings would use conventional ACI 318.

### 5.5 Rad-Hardened HVAC Systems

**What it means:** Nuclear HVAC systems must:
- Maintain negative pressure in radiological areas (confinement)
- Use HEPA filtration (99.97% at 0.3 micron) on exhaust
- Use redundant trains for safety-related ventilation
- Survive design basis accidents (seismic, tornado missiles)
- Be NQA-1 qualified
- Include continuous air monitoring (CAMs)

**Cost impact:**
- Conventional commercial HVAC: ~$15-40/sq ft of served area
- Nuclear safety-related HVAC: estimated **3-5x** conventional due to redundancy, NQA-1 qualification, HEPA banks, hardened ductwork, and monitoring
- Implied nuclear HVAC: ~$50-200/sq ft of served area

**Fusion relevance:** DT plants need rad-HVAC for:
- Tritium processing areas (confinement zones)
- Hot cell
- Reactor hall (tritium leak containment)
pB11 plants: conventional HVAC everywhere.

### 5.6 Summary: Nuclear-Grade Cost Multipliers

| Construction Element | Nuclear/Industrial Multiplier | Confidence | Key Driver |
|---------------------|---:|---:|-----------|
| Concrete (material) | 1.5x | High | Tighter specs, traceability |
| Concrete (installed) | 2.0-3.0x | Medium | Slower installation, QA, rebar density |
| Structural steel | 2.2x | Medium | Material premium + 41% QA overhead |
| Mechanical components | 5-50x | Low (wide range) | NQA-1 procurement, documentation |
| HVAC systems | 3-5x | Medium | Redundancy, HEPA, hardening, NQA-1 |
| Electrical (safety-related) | 3-10x | Medium | Qualification, redundancy, separation |
| **Overall structure (blended)** | **2.5-4.0x** | Medium | Blend of all above |

The blended multiplier of **2.5-4.0x** is consistent with the observed ratio of nuclear to CCGT building costs ($800-1,300/kW vs $150-250/kW = 3.5-6x), noting that nuclear plants also have more and larger buildings than CCGTs.

---

## 6. Hot Cell Construction Costs

### What is a Hot Cell?
A shielded enclosure for handling highly radioactive materials remotely. Features:
- 3-4 foot thick high-density reinforced concrete walls (or equivalent shielding)
- Stainless steel interior liner
- Leaded glass or zinc bromide shielding windows
- Remote manipulators (master-slave or robotic)
- Independent HVAC with HEPA filtration
- Fire suppression
- Contamination control systems
- Full NQA-1 QA program (for nuclear facilities)

### Hot Cell Cost Data Points

| Facility | Cost | Year | Size/Scope | Grade |
|----------|-----:|---:|-----------|-------|
| SHINE Medical Isotope Facility (Janesville, WI) | ~$100M total facility | 2019-2024 | 91,000 sq ft total, 3 hot cells for Mo-99 | NRC Part 30 licensed |
| ORNL Stable Isotope Production & Research Center (SIPRC) | $325M total project ($88.8M construction contract) | 2025-2027 | 64,000 sq ft, single story | DOE nuclear facility |
| ORNL Radiochemical Engineering Development Center | N/A (legacy) | 1960s | 15 hot cells | DOE nuclear facility |
| ITER Hot Cell Complex | Estimated EUR 200-500M (portion of total) | Under construction | Multiple buildings, school-bus-sized components, 45-tonne handling | Nuclear (INB), seismic qualified |
| DOE pit production hot cells (LANL) | Billions (part of larger complex) | Ongoing | Weapons-grade, extreme security | DOE defense nuclear |

### Hot Cell Cost per Square Foot (estimated)
- **Basic medical isotope hot cell:** $1,000-2,000/sq ft (based on SHINE facility scale)
- **DOE research hot cell:** $2,000-5,000/sq ft (based on SIPRC and comparable facilities)
- **Large nuclear hot cell (ITER-class):** $3,000-8,000/sq ft (based on scale and nuclear QA)
- **Defense nuclear hot cell:** $5,000-15,000+/sq ft

### Fusion Hot Cell Requirements
For a DT fusion plant, the hot cell handles:
- Activated first-wall and blanket components
- Contaminated (tritiated) components
- Size reduction, decontamination, packaging for disposal
- Remote cutting, welding, inspection

The current model estimates hot cell at **$93.4/kW** (= ~$109M for a 1.165 GWe plant). At an assumed hot cell footprint of ~5,000-10,000 sq ft, this implies **$10,900-21,800/sq ft** -- which is at the high end but within range for a purpose-built remote-handling hot cell at a nuclear facility.

For pB11 fuel (no activation, no tritium): hot cell is **not needed**, scaled to 0.5x in the current model. A more aggressive scaling (0.1-0.2x) may be justified since pB11 produces negligible activation products and no tritium.

---

## 7. Implications for Fusion Building Cost Model

### DT Fusion Plant (Part 30 Licensed)
DT fusion falls under NRC Part 30 (byproduct materials license for tritium), NOT Part 50 (reactor license). This means:
- **No Seismic Category I** requirement (use IBC/ASCE 7)
- **No fission containment** (confinement approach instead -- sealed building, negative pressure, filtered exhaust)
- **NQA-1 only for tritium systems** -- not the whole plant
- **No ACI 349 nuclear concrete** for most buildings -- use ACI 318
- **Rad-HVAC required** only in tritium zones and hot cell

Expected building construction grade: **Enhanced industrial** with selective nuclear-grade elements.

Cost multiplier vs. pure industrial: **~1.3-1.8x** (weighted average, considering most buildings are industrial with some nuclear-grade subsystems).

### pB11 Fusion Plant (Industrial Grade)
pB11 produces no tritium and negligible neutron activation. Regulatory posture:
- **No NRC license** required (no byproduct material)
- **Pure industrial construction** throughout
- **No hot cell** needed
- **No rad-HVAC** needed
- **Conventional concrete** (ACI 318) everywhere
- **Standard commercial HVAC**

Expected building construction grade: **Pure industrial**, comparable to a large gas turbine or heavy manufacturing plant.

Cost multiplier vs. pure industrial: **~1.0x**

### Recommended Cost Ranges for Model Validation

| Building Category | CCGT Benchmark | DT Fusion | pB11 Fusion | Nuclear Fission |
|------------------|---:|---:|---:|---:|
| Site improvements | $50-100/kW | $200-300/kW | $50-100/kW | $200-400/kW |
| Main equipment building | $50-100/kW | $100-150/kW | $80-120/kW | $300-500/kW |
| Turbine building | $30-60/kW | $40-70/kW | $40-70/kW | $50-100/kW |
| Hot cell | N/A | $70-120/kW | $0-10/kW | $50-150/kW |
| Control room | $10-20/kW | $15-25/kW | $10-20/kW | $30-60/kW |
| Balance of buildings | $30-50/kW | $80-120/kW | $40-60/kW | $150-300/kW |
| **Total buildings** | **$150-250/kW** | **$500-800/kW** | **$200-400/kW** | **$800-1,300/kW** |

### Current Model Assessment
The current CAS21 model values:
- **DT: $760/kW** -- within the $500-800/kW range. Reasonable.
- **Non-DT: $511/kW** -- above the $200-400/kW range for pB11. The 0.5x scaling may be **insufficiently aggressive** for truly aneutronic fuel cycles. The site improvements line ($134/kW for non-DT) is still high for a plant that needs no nuclear-specific site infrastructure.

### Potential Model Refinements
1. **Site improvements ($268/kW DT)** is the largest single line item and appears high. CCGT site prep is $50-100/kW. Even with fusion-specific infrastructure (cryo lines, magnet power distribution, large equipment pads), $150-200/kW may be more appropriate. The current value may be carrying costs that belong in other accounts (utility distribution, emergency systems).

2. **Hot cell scaling for pB11** should be 0.0-0.1x, not 0.5x. A pB11 plant with negligible activation needs no hot cell -- perhaps only a small shielded inspection area.

3. **Fuel storage scaling for pB11** should approach zero (no tritium to store).

4. **Consider a 3-tier model:** DT (1.0x) / DD-DHe3 (0.6-0.7x) / pB11 (0.3-0.4x) rather than binary DT/non-DT.

---

## Sources

### CCGT Costs
- [Thunder Said Energy - Gas Power Economics](https://thundersaidenergy.com/downloads/gas-to-power-project-economics/)
- [Gas Turbine World - Capital Costs](https://gasturbineworld.com/capital-costs/)
- [EIA - Construction Costs for Power Plants](https://www.eia.gov/todayinenergy/detail.php?id=31912)
- [Gas Turbine EPC Costs 2025](https://www.uspeglobal.com/blog/76774-gas-turbine-epc-costs-2026-complete-breakdown-by-project-size-region-fuel-type)

### Coal Plant Costs
- [NETL Cost and Performance Baseline (DOE/NETL-2015/1723)](https://www.env.nm.gov/wp-content/uploads/sites/2/2019/12/AQBP-CostandPerformanceBaselineforFossilEnergyPlants_DOE_NETL-2015-1723.pdf)
- [NETL Baseline Tool](https://netl.doe.gov/NETLVol1BaselineTool)
- [IDAES NETL Costing Documentation](https://idaes-pse.readthedocs.io/en/stable/reference_guides/model_libraries/power_generation/costing/power_plant_costing_netl.html)

### Nuclear Plant Costs
- [MIT CANES - Overnight Capital Cost of Next AP1000](https://web.mit.edu/kshirvan/www/research/ANP193%20TR%20CANES.pdf)
- [MIT CANES - 2024 Total Cost Projection](https://web.mit.edu/kshirvan/www/research/ANP201%20TR%20CANES.pdf)
- [NREL ATB 2024 - Nuclear](https://atb.nrel.gov/electricity/2024/nuclear)
- [INL SFR Cost Study (Sort_67398)](https://inldigitallibrary.inl.gov/sites/sti/sti/Sort_67398.pdf)
- [INL Meta-Analysis of Reactor Costs (RPT-24-77048)](https://gain.inl.gov/content/uploads/4/2024/11/INL-RPT-24-77048-Meta-Analysis-of-Adv-Nuclear-Reactor-Cost-Estimations.pdf)
- [INL Literature Review of Advanced Reactor Costs](https://gain.inl.gov/content/uploads/4/2024/11/INL-RPT-23-72972-Literature-Review-of-Adv-Reactor-Cost-Estimates.pdf)
- [Stewart 2022 MIT Thesis - Capital Cost Evaluation](https://dspace.mit.edu/bitstream/handle/1721.1/144869/Stewart-wstewart-phd-nse-2022-thesis.pdf?sequence=1)
- [IFP - Why Does Nuclear Cost So Much](https://ifp.org/nuclear-power-plant-construction-costs/)
- [INL Cost Reduction for AP1000](https://sai.inl.gov/content/uploads/29/2025/06/M3_SAI-AP1000_Lessons_Rev6-nocomments-002.pdf)

### Nuclear-Grade Construction Cost Drivers
- [Breakthrough Institute - To Cut Nuclear Costs, Cut Concrete](https://thebreakthrough.org/issues/energy/to-cut-nuclear-costs-cut-concrete)
- [Construction Physics - Why Nuclear Costs So High Part II](https://www.construction-physics.com/p/why-are-nuclear-power-construction-370)
- [Construction Physics - Why Nuclear Costs So High Part I](https://www.construction-physics.com/p/why-are-nuclear-power-construction)
- [Joule - Sources of Cost Overrun in Nuclear Construction](https://www.cell.com/joule/fulltext/S2542-4351(20)30458-X)
- [ORNL - Making Nuclear Construction More Affordable](https://www.ornl.gov/news/how-ornl-research-making-nuclear-reactor-construction-more-affordable)
- [DOE - Commercial Grade Dedication Guidance](https://www.energy.gov/sites/prod/files/em/CommercialGradeDedicationGuidance.pdf)
- [NRC - Seismic Design Classification (RG 1.29)](https://www.nrc.gov/docs/ML2115/ML21155A003.pdf)
- [OECD/NEA - Economic Effect of Seismic Load on Nuclear Costs](https://www.sciencedirect.com/science/article/abs/pii/0029549378902182)

### Industrial Construction Benchmarks
- [Cushman & Wakefield - Industrial Construction Cost Guide](https://www.cushmanwakefield.com/en/united-states/insights/industrial-construction-cost-guide)
- [RS Means - Factory 1-Story Cost Model](https://www.rsmeans.com/model-pages/factory-1-story)
- [Cushman & Wakefield - Data Center Cost Guide](https://www.cushmanwakefield.com/en/united-states/insights/data-center-development-cost-guide)

### Hot Cell Facilities
- [SHINE Medical Isotope Production Facility](https://www.pharmaceutical-technology.com/projects/shine-medical-isotope-production-facility/)
- [ORNL SIPRC - DOE Awards $88.8M Contract](https://www.isotopes.gov/department-energy-awards-88m-contract-build-stable-isotope-production-and-research-facility)
- [ITER Hot Cell Overview](https://www.iter.org/machine/supporting-systems/hot-cell)
- [ITER Hot Cell Complex Introduction](https://indico.iter.org/event/14/attachments/95/140/Introduction_to_the_ITER_Hot_Cell_Complex.pdf)
- [Wikipedia - Hot Cell](https://en.wikipedia.org/wiki/Hot_cell)

### Fusion Cost Studies
- [ARIES Cost Account Documentation (UCSD-CER-13-01)](https://qedfusion.org/LIB/REPORT/ARIES-ACT/UCSD-CER-13-01.pdf)
- [Kairos Power - Hermes Construction](https://www.energy.gov/ne/articles/kairos-power-starts-construction-hermes-reactor)
- [NRC - Tritium Handling Facility Evaluation](https://www.nrc.gov/docs/ML1029/ML102990087.pdf)

### Advanced Reactor Construction Approaches
- [Kairos Power - Hermes Reactor](https://kairospower.com/external_updates/kairos-power-begins-nuclear-construction-of-hermes-demonstration-reactor/)
- [Breakthrough Institute - How to Make Nuclear Cheap](https://s3.us-east-2.amazonaws.com/uploads.thebreakthrough.org/legacy/images/pdfs/Breakthrough_Institute_How_to_Make_Nuclear_Cheap.pdf)
