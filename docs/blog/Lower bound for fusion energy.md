# The Lower Bound for Fusion Energy Cost

If the deuterium-tritium fusion core were free — every magnet, every blanket, every heating system, handed to you at zero cost — the plant would still produce electricity at around  $29/MWh. That is nearly three times the [1-cent per kWh target](https://1cf.energy/). The cost uncertainty is not in the turbines or the buildings. It is in the fusion core. And the range of estimates for that core is enormous. This dispatch will examine the theoretical cost-of-electricity floor in a power plant where the fusion core, the largest unknown, is ignored.

Published estimates for a commercial fusion plant range from [$47/MWh](https://www.sciencedirect.com/science/article/abs/pii/S0920379605007210) for an optimistic nth-of-a-kind advanced tokamak (ARIES-AT, Najmabadi et al., 2006) to [$160+/MWh](https://www.sciencedirect.com/science/article/pii/S0360544218305395) for a first-generation EU-DEMO (Entler et al., 2018), with most serious estimates [landing between $50 and $130/MWh](https://www.tandfonline.com/doi/abs/10.13182/FST15-157) (Sheffield & Milora, 2016). Recent analysis from Cambridge [suggests early fusion plants will exceed $150/MWh](https://www.sciencedirect.com/science/article/abs/pii/S0301421523000964) even with production learning (Lindley et al., 2023).

Cost estimates for fusion projects are kept under wraps, and even experts have a hard time estimating the eventual costs. A [recent Nature Energy analysis](https://www.nature.com/articles/s41560-026-02023-8) compiled first-of-a-kind CAPEX estimates from experts and the literature and found a range of **$1,400 to $43,000 per kilowatt** — a 30x spread (Tang et al., 2026). That study determined that the experience rates for a deuterium-tritium tokamak fusion core would likely end up in the range of 2-8% cost reduction per doubling of deployment. This revises the 8-20% estimate assumed previously. These rates are a result of the size and complexity of the technology.

A fusion plant has to convert the energy generated in the fusion core to useful electricity. For this purpose, most projects envision using a heat engine — turbines, generators, cooling towers — in addition to a switchyard, a control room, and a building to put it all in. These components are mature industrial hardware, with no real price uncertainty or learning rates. Beyond the equipment itself, a plant incurs indirect costs (engineering, project management), owner's costs (staff recruitment and training), supplementary costs (shipping, spares, insurance, decommissioning provisions), financing charges, and ongoing operations and maintenance. All of these are well-characterized from existing power plant experience.

We are seeking corridors for fusion to have a truly revolutionary price point — 1 cent per kilowatt-hour. If we start by accounting for the components we're certain about and that exist today at commercial scale, we could find the lower bound on what fusion electricity can cost.

To find the lower bound on fusion energy cost we can ask: what is the cost of the plant if the fusion core were free?

## The Heat Engine — Generating Electricity from Heat

The main product of a fusion core is heat, and every fusion plant has to convert that heat into electricity. Most designs envision a thermal cycle — steam Rankine or supercritical CO2 Brayton. In either case, the plant needs buildings, electrical infrastructure, a grid connection, a control room, and staff.

Using the [1costingfe](https://github.com/1cfe/1costingfe) techno-economic model, we can price these components for a 1 GWe plant. The table below uses a supercritical CO2 Brayton cycle, which gives the lowest floor of the three options we modeled (Rankine, sCO2, and combined cycle). All numbers in this post can be reproduced with the [companion script](https://github.com/1cfe/1costingfe/blob/master/examples/lower_bound_blog_numbers.py).

The power conversion equipment is the same regardless of fuel choice:

| Component | Cost | What it is |
| --- | --- | --- |
| Turbine & generator | $180M | sCO2 turbomachinery, comparable to gas plant hardware |
| Electrical plant | $97M | Switchgear, transformers, grid connection |
| Miscellaneous equipment | $59M | Cranes, compressed air, fire protection |
| Heat rejection | $25M | Cooling towers and circulating water (smaller with higher efficiency) |
| **BOP subtotal** | **$360M** |  |

## The Rest of the Plant

Buildings are not fuel-independent. A fusion plant includes roughly 18 distinct buildings and site structures — reactor building, turbine hall, heat exchanger building, cryogenics facility, power supply building, control room, maintenance shops, and others. About half of these (turbine hall, cryogenics, switchgear, assembly hall) are the same regardless of fuel. The other half — reactor building, hot cell, ventilation systems, fuel storage, site improvements — vary with the radiation and tritium hazards of the fuel. A DT plant needs biological shielding walls in the reactor building, a hot cell for remote handling of activated components (\$90M by itself), tritium containment barriers, and nuclear-rated HVAC with HEPA filtration. An aneutronic pB11 plant needs none of these — the reactor building is a standard heavy industrial crane hall, there is no hot cell, and the HVAC is conventional. The result is a range from \$320M (pB11, industrial-grade) to \$570M (DT, enhanced-industrial). We return to the fuel-specific breakdown below.

The BOP equipment is priced from [NETL's Cost and Performance Baseline for Fossil Energy Plants](https://www.osti.gov/servlets/purl/1893822) (DOE/NETL-2022/3575) and the [ARIES cost account documentation](https://cer.ucsd.edu/_files/publications/UCSD-CER-13-01.pdf) (Waganer, UCSD-CER-13-01, 2013), adjusted to 2024 dollars.

On top of the direct costs, there are indirect costs (engineering, project management — roughly 20% of directs), owner's costs, supplementary costs (shipping, spares, insurance, decommissioning provisions), and financing charges. A plant also needs staff — roughly 30 to 80 full-time employees at 1 GWe scale depending on fuel choice and the associated radiation protection requirements, costing $18-40M/year in loaded operations and maintenance (O&M). All of these costs are included in the LCOE floors reported below.

## The Balance of Plant for Deuterium-Tritium

Deuterium-tritium (DT) is the mainstream fuel choice for fusion. It produces 80% of its energy as 14.1 MeV neutrons, which activate structural components and require heavy shielding, and it uses radioactive tritium — an environmental hazard if it leaks into groundwater. Even with a free fusion core, a DT plant still needs buildings with radiation shielding, tritium containment with secondary barriers, hot cells for remote handling of activated components, and nuclear-rated HVAC. These requirements are driven by the hazards themselves — neutron activation and tritium inventory — not by regulatory classification. The buildings cost $570M at 1 GWe, and the plant requires roughly 80 staff for radiation protection, tritium handling, and the maintenance procedures these hazards demand.

For a 1 GWe DT tokamak at standard financial assumptions (7% weighted average cost of capital (WACC), 85% availability, 30-year life, 6-year construction), the Levelized Cost of Electricity (LCOE) floor with a free fusion core is **$29/MWh** — nearly three times the 1-cent target ($10/MWh). The overnight capital cost of this core-free plant is $1,700/kW.

For context, the fully costed DT tokamak at the same conditions comes in at $82/MWh. The fusion core accounts for about two-thirds of that. But the remaining third — $29/MWh of buildings, BOP, staffing, and financing — already exceeds the 1-cent target by itself. No improvement in the fusion core can close a gap that lives outside the core.

## Squeezing the DT Floor

The $29/MWh floor assumes a single 1 GWe plant built with commercial project finance. Can we push it down to $10/MWh by improving the parameters that are independent of the fusion core? The parameters are:

- **Scale** spreads fixed costs (buildings, staff, indirects) over more megawatt-hours
- **Availability** produces more energy per dollar of installed capital
- **Financing cost (WACC)** determines the annual capital charge rate
- **Construction time** determines interest during construction
- **Plant lifetime** determines how many years of revenue amortize the capital

| Scenario (DT) | Floor (\$/MWh) | Overnight (\$/kW) | Budget left for core |
| --- | --- | --- | --- |
| Baseline: 1 GWe, 85%, 7% WACC, 30yr, 6yr build | 29 | 1,700 | -\$19/MWh |
| 2 GWe, 85%, 7%, 30yr, 6yr | 23 | 1,500 | -\$13/MWh |
| 2 GWe, 95%, 3% WACC, 50yr, 3yr build | 14 | 1,200 | -\$3.6/MWh |
| 3 GWe, 95%, 3%, 50yr, 3yr | 12 | 1,100 | -\$1.8/MWh |
| 5 GWe, 95%, 2% WACC, 50yr, 3yr | 9.5 | 1,100 | +\$0.5/MWh |

Even at the most aggressive conditions — a 5 GWe plant with 95% availability, 2% WACC, 50-year life, and 3-year construction — the DT floor barely crosses $10/MWh, leaving only $0.5/MWh of budget for the fusion core. That translates to roughly $50/kW of overnight capital for the entire core: magnets, heating, vacuum vessel, structure, power supplies, and installation. For reference, a single large superconducting magnet costs more than this. The DT floor cannot be squeezed far enough to make room for the core.

The reason is the buildings and staffing. DT buildings cost $570M at 1 GWe because neutrons and tritium demand shielding, hot cells, containment barriers, and nuclear-rated HVAC. DT staffing runs to 78 FTE because those hazards require health physics technicians, tritium processing personnel, radwaste handlers, and hot cell operators. These costs scale down with plant size but never disappear — they are structural consequences of the fuel choice, not of the plant design.

## Squeezing DT Staffing

At the most aggressive conditions in the table above (5 GWe, 2% WACC), the DT floor is $9.5/MWh. O&M staffing accounts for roughly half of that. DT staffing is high — 78 FTE at 1 GWe — because of the radiation and tritium hazards: health physics technicians (10-15), tritium processing and accountability personnel (10-20), radioactive waste handling (5-10), and hot cell operators for remote maintenance of activated components. Can automation close the remaining gap?

Lights-out operation — already deployed in semiconductor fabs, chemical plants, and automated factories — could reduce a conventional thermal plant from 30 to 15 full-time employees. But a DT plant is not a conventional thermal plant. The radiation-specific roles resist automation differently: hot cell remote handling is already robotic, but the programming, supervision, and exception handling still require humans. Tritium accountability is a regulatory function. Health physics coverage is required whenever personnel enter controlled areas — and someone has to enter for the maintenance that remote handling cannot reach.

What prevents going to zero? The remaining functions are grid dispatch (mostly SCADA-automated already), routine maintenance, unplanned fault response, regulatory compliance, and security. Routine operations are automatable with current technology. The binding constraint is the unplanned event: a novel failure mode that requires judgment under uncertainty. If the 10-year technology trajectory holds — humanoid robotics for physical intervention, AI agents matching human expert judgment in relevant domains — the binding constraint shifts from technical capability to regulatory accountability. Utility-scale grid infrastructure may still require human accountability for some period regardless of what the technology can do. That is likely the imposed floor on staffing, not what is technically possible.

Even with radical automation, DT retains a staffing penalty. The radiation-specific roles may be reducible but not eliminable on any near-term horizon — as long as humans enter the plant for any reason, health physics coverage is required; as long as tritium is on site, accountability is required. Optimistically halving DT staffing from 78 to 39 FTE at the aggressive 5 GWe conditions might shave $1-2/MWh from the floor, bringing it to $8-9/MWh. That is still not enough headroom for the core.

## Fuel Choice Reshapes the Floor

The DT floor resists every lever we have applied — scale, financing, construction time, and staffing automation. The constraint is not the plant parameters. It is the fuel.

Different fuel choices produce different floors.

| Fuel | Buildings (1 GWe) | BOP floor (excl. fuel) | Staffing |
| --- | --- | --- | --- |
| DT | $570M | $29/MWh | 78 FTE |
| D-He3 | $380M | $19/MWh | 39 FTE |
| pB11 | $322M | $17/MWh | 36 FTE |

A pB11 plant needs roughly the same staff as a conventional thermal plant plus fusion-specific roles (magnet technicians, vacuum systems, plasma control) — but none of the radiation-specific roles that dominate DT staffing. No health physics, no tritium processing, no radwaste, no hot cell operators.

Deuterium-helium-3 (D-He3) produces about 5% of its energy as neutrons from DD side reactions — far less than DT, but not zero. The buildings require some radiation shielding and modest tritium monitoring, bringing them to $380M. The BOP floor drops to $19/MWh. However, D-He3 carries a separate problem: helium-3 fuel at current prices ($2M/kg) contributes $79/MWh to the levelized electricity cost, pushing the total free-core floor to $98/MWh. The BOP is competitive; the fuel supply is not.

Proton-boron (pB11) fuel is aneutronic — 99.8% of its fusion energy comes out as charged alpha particles, not neutrons. It uses no tritium and does not activate structural components. The buildings can be built to conventional industrial standards — no shielding, no hot cells, no tritium containment infrastructure. The result is a BOP floor of **$17/MWh**, roughly half the DT floor.

The $250M building cost gap between DT and pB11 is larger than most fusion core cost-reduction scenarios in the literature. This gap is not a statement about plasma physics difficulty. It is a statement about what the expectation of handling neutrons and tritium does to the cost of the building you put the plant in. The fuel choice reshapes the floor before the fusion core enters the picture.

## Squeezing the pB11 Floor

The pB11 baseline floor is $17/MWh — still 1.7x the target, but much closer than DT. Applying the same levers:

| Scenario (pB11) | Floor (\$/MWh) | Overnight (\$/kW) | Budget left for core |
| --- | --- | --- | --- |
| Baseline: 1 GWe, 85%, 7% WACC, 30yr, 6yr build | 17 | 1,100 | -\$7/MWh |
| 2 GWe, 85%, 7%, 30yr, 6yr | 14 | 970 | -\$4/MWh |
| 2 GWe, 95%, 3% WACC, 50yr, 3yr build | 7.3 | 770 | +\$2.7/MWh |
| 3 GWe, 95%, 3%, 50yr, 3yr | 6.4 | 730 | +\$3.6/MWh |
| 5 GWe, 95%, 2% WACC, 50yr, 3yr | 5.1 | 690 | +\$4.9/MWh |

At 2 GWe with 95% availability, 3% WACC, 3-year construction, and 50-year life, the pB11 floor drops to $7/MWh — below the target — leaving $2.7/MWh of budget for the fusion core. Compare this to DT at the same conditions: $14/MWh floor, no budget for the core at all.

Is 3-year construction realistic for the BOP? Modern gas combined-cycle plants of comparable scale are built in 2-3 years. The BOP of a fusion plant is similar in scope and complexity. In the scenario we are solving for, we do not construct the fusion core, only the turbine halls, cooling systems, switchyards, and buildings. A 3% WACC is not realistic for an unsubsidized first-of-a-kind plant — commercial project finance for new energy technologies typically runs 7-10%. But government-backed financing (DOE Loan Programs Office, export credit agencies) routinely achieves 2-4% for qualifying infrastructure. The Vogtle nuclear expansion, for example, received an \$8.3B DOE loan guarantee. Reaching 3% WACC assumes the plant is financed as infrastructure, not as a venture bet.

At the 2 GWe aggressive conditions, the remaining budget for the fusion core is $3/MWh — roughly $850/kW of overnight capital for magnets, heating, vacuum vessel, structure, power supplies, and installation. For reference, the entire overnight cost of a natural gas combined-cycle plant is $900-1,200/kW. The fusion core has to fit inside that budget while being none of those things — mature, mass-produced, or built at scale — yet.

O&M staffing accounts for \$4/MWh of the \$7/MWh pB11 floor — more than half. Because pB11 has no radiation-specific roles, the staffing automation argument is much stronger here. The remaining functions — grid dispatch, routine maintenance, unplanned fault response — are the same as any conventional thermal plant. Lights-out operation could bring staffing from 30 to 15 employees, dropping the floor from \$7 to \$5/MWh.

Radical staffing reduction also reshapes the buildings. A 15-FTE plant still needs a control room, canteen, locker rooms, parking. A near-zero-FTE plant looks more like a data center: remote monitoring, automated systems, minimal human-occupied space. That could strip \$50-100M from the \$320M building cost, flowing directly into overnight capital and LCOE. Together, near-zero staffing and the associated building scope reduction would drop the floor from \$7/MWh to roughly \$4/MWh — leaving \$6/MWh of budget for the fusion core, more than double the baseline. At the limit, automation does not just reduce O&M — it removes building scope.

## Deleting the Floor

Everything above assumes a thermal cycle: fusion produces heat, a turbine converts heat to electricity. The turbine, cooling towers, and the buildings that house them are the floor. But for aneutronic fuels, this assumption is optional.

Proton-boron fusion produces 99.8% of its energy as charged alpha particles — not heat. These particles can, in principle, be converted directly to electricity without a thermal cycle. Direct energy conversion bypasses the turbine, the cooling towers, and much of the building. It does not lower the floor — it removes it.

This is not a theoretical curiosity. Helion Energy's approach to D-He3 fusion is predicated on direct energy conversion: pulsed field-reversed configurations that capture fusion energy inductively, without a steam cycle. If direct conversion works at the efficiencies its proponents project, the BOP analysis in this post becomes largely irrelevant — the plant is the core, a power conditioning system, and a grid connection.

Direct energy conversion is less mature than thermal cycles, and its cost uncertainty is much larger. The efficiency, capital cost, and reliability of direct conversion at utility scale are open questions. We will explore the economics of direct conversion in a future post.

## Conclusions

**1. DT cannot reach 1 cent.** Even with a free fusion core, a 5 GWe DT plant at the most aggressive financial conditions barely crosses $10/MWh, leaving essentially no budget for the core. Even radical staffing automation cannot close the gap — radiation-specific roles resist elimination. The buildings ($570M), staffing (78 FTE), and the procedures required to handle neutrons and tritium create a floor that is a structural consequence of the fuel, not the plant design.

**2. Fuel choice is a BOP decision, not just a core decision.** The $250M building cost gap between DT ($570M) and pB11 ($320M) is larger than most fusion core cost-reduction scenarios. Aneutronic fuel downgrades the buildings from enhanced-industrial to industrial, eliminates scheduled replacement campaigns for the fusion core, and removes tritium infrastructure entirely. These savings compound through indirect costs and financing. The DT floor ($29/MWh) is nearly double the pB11 floor ($17/MWh). D-He3 falls in between on BOP ($19/MWh) but is burdened by helium-3 fuel costs.

**3. No single lever reaches 1 cent.** At standard financial conditions, even the pB11 LCOE floor is 1.7x the target with a free fusion core. Reaching $10/MWh requires at least four favorable conditions simultaneously: scale (2+ GWe), high availability (95%+), low-cost financing (3% WACC or below), and fast construction (3 years or less). Each of these is individually achievable. Together they represent a systems integration challenge as much as an engineering one.

**4. Scale matters more than unit cost.** Going from 1 GWe to 2 GWe buys more LCOE reduction than any other single lever. Buildings, staff, and indirect costs do not double when the plant doubles — staffing scales as P^0.5, so a 2 GWe plant needs 1.4x the staff of a 1 GWe plant, not 2x. This economy of scale does not depend on any technology breakthrough.

**5. Direct conversion could delete the floor entirely.** For aneutronic fuels, direct energy conversion bypasses the thermal cycle — the largest single component of the floor. If it works at the projected efficiencies, the economics of fusion change category: from squeezing a floor to building a fundamentally different kind of power plant.

## The Path Forward

The lower bound is a useful limit case. No fusion core will be free. But the exercise reveals where the constraints lie: not in the fusion core, but in the industrial cost structure around it.

A pB11 fusion plant at 2 GWe scale, with 95% availability, 3% cost of capital, 3-year construction, and 50-year life, has a balance-of-plant floor of $7/MWh and a fusion core budget of $850/kW. That budget is tight but not zero, and it grows with scale.

Reaching 1-cent fusion energy is a whole-plant problem. It requires large plants, high availability, fast construction, low-cost financing, long plant life, and a fuel that does not burden the balance of plant with radioactivity safety requirements. Every component of this path — industrial buildings, large turbines, high-availability operations, government-backed financing, long-lived civil structures — exists today, in other industries, at the required scale.

Tang et al.'s Nature Energy analysis arrives at a compatible conclusion from the opposite direction: mainstream fusion designs have technological characteristics — enormous unit size, extraordinary complexity, heavy customization — that predict slow cost learning. Their recommendation is to explore alternative fusion concepts with more favorable attributes, including smaller unit size, reduced complexity, and alternative fuels like proton-boron. The lower-bound analysis shows why: simpler, smaller, aneutronic designs have a lower floor to build on, not just a better shot at a cheap fusion core.

The fusion core is the hard part. The floor tells us how much room it has to work with.

## References

1. Najmabadi, F. et al. "The ARIES-AT advanced tokamak, Advanced technology fusion power plant." *Fusion Engineering and Design*, 80, 3-23 (2006). [Link](https://www.sciencedirect.com/science/article/abs/pii/S0920379605007210)
2. Entler, S. et al. "Approximation of the economy of fusion energy." *Energy*, 152, 489-497 (2018). [Link](https://www.sciencedirect.com/science/article/pii/S0360544218305395)
3. Sheffield, J. & Milora, S. "Generic Magnetic Fusion Reactor Revisited." *Fusion Science and Technology*, 70(1), 14-35 (2016). [Link](https://www.tandfonline.com/doi/abs/10.13182/FST15-157)
4. Lindley, B. A. et al. "Can fusion energy be cost-competitive and commercially viable? An analysis of magnetically confined reactors." *Energy Policy*, 177 (2023). [Link](https://www.sciencedirect.com/science/article/abs/pii/S0301421523000964)
5. Tang, L. et al. "Fusion power experience rates are overestimated." *Nature Energy* (2026). [Link](https://www.nature.com/articles/s41560-026-02023-8)
6. U.S. DOE/NETL. "Cost and Performance Baseline for Fossil Energy Plants, Volume 1." DOE/NETL-2022/3575 (2022). [Link](https://www.osti.gov/servlets/purl/1893822)
7. Waganer, L. "ARIES Cost Account Documentation." UCSD-CER-13-01 (2013). [Link](https://cer.ucsd.edu/_files/publications/UCSD-CER-13-01.pdf)
8. 1cFE. "1costingfe: Open-source fusion techno-economic model." [GitHub](https://github.com/1cfe/1costingfe)