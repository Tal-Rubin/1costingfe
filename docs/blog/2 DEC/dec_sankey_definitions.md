# Sankey Diagram Definitions for "Direct Energy Conversion and the Cost Floor"

Generated using [SankeyMatic](https://sankeymatic.com/build/). Paste each block into the editor.

All energy values in MW. All cases are 1 GWe net output, baseline conditions
(85% availability, 7% WACC, 30-year life, 6-year construction).

Power balance from `1costingfe` model (mirror concept, f_rad_fus=0.83 for p-B11
with alpha channeling per Ochs et al. 2022, f_rad_fus=0.25 for D-He3).

"Heating System." (with period) is a SankeyMatic trick to create a separate node
from the left-side "Heating System". After export, edit the SVG to reroute the
flow back to the left node, remove the period, and move the label into the loop.

## Figure 1: p-B11 Thermal Cycle

Placement: end of "What Direct Energy Conversion Does"

p_fus=2397, p_rad=1990 (83%), transport=448, p_th=2438, p_the=1146

```
// p-B11 thermal only (f_dec=0), 1 GWe net output
// 83% of fusion power radiated as bremsstrahlung
// Heating: 80 MW wall-plug, 40 MW delivered to plasma

Fusion Reactions [2397] Charged Particles
Heating System [40] Charged Particles

// Bremsstrahlung dominates: 1990 MW radiated, 448 MW transport
Charged Particles [1990] Bremsstrahlung → Thermal (sCO2)
Charged Particles [448] Transport → Thermal (sCO2)

Bremsstrahlung → Thermal (sCO2) [935] Gross Electric
Bremsstrahlung → Thermal (sCO2) [1055] Waste Heat

Transport → Thermal (sCO2) [211] Gross Electric
Transport → Thermal (sCO2) [237] Waste Heat

Gross Electric [1000] Net Electric
Gross Electric [106] BOP
Gross Electric [40] Heating System.

:Fusion Reactions #d45
:Charged Particles #2a7fff
:Bremsstrahlung → Thermal (sCO2) #e8a030
:Transport → Thermal (sCO2) #888
:Gross Electric #6a3
:Net Electric #0a0
:Waste Heat #c33
:BOP #bcbd22
:Heating System #c80
:Heating System. #c80
```

## Figure 2: D-He3 Pulsed Inductive DEC 85% — textbook 50/50 reference

Placement: end of "Pulsed Inductive DEC", as the textbook reference paired
with Figure 4 (Helion-likely operating point).

Textbook reference: 50/50 D/³He mix at T=70 keV with asymptotic T burnup
(f_T = 0.97, no He-3 recovery, no Helion-style D-rich shift). Bremsstrahlung
fraction f_brem = 0.242 from Bosch-Hale + relativistic-bremsstrahlung
(see `examples/dhe3_mix_optimization.py`).

p_fus=1535, p_ash=1442, p_neutron=92, p_rad=371 (24%), transport=1111,
DEC gets 95% of transport=1055, thermal gets brem+neutrons+5% transport.

```
// D-He3 pulsed inductive DEC at 85%, textbook 50/50 reference
// f_DD=0.131, f_T=0.97, f_He3=0 (no recovery), f_brem=0.242
// 95% of charged transport to DEC, 5% + brem + neutrons to thermal

Fusion Reactions [1442] Charged Particles
Fusion Reactions [92] Neutrons
Heating System [40] Charged Particles

// Transport = 1442 + 40 - 371 = 1111
// 95% to DEC = 1055, 5% to walls = 56
Charged Particles [371] Bremsstrahlung → Thermal (sCO2)
Charged Particles [1055] Pulsed Inductive DEC
Charged Particles [56] Thermal (sCO2)
Neutrons [92] Thermal (sCO2)

Pulsed Inductive DEC [897] Gross Electric
Pulsed Inductive DEC [158] Waste Heat

Bremsstrahlung → Thermal (sCO2) [174] Gross Electric
Bremsstrahlung → Thermal (sCO2) [197] Waste Heat

Thermal (sCO2) [70] Gross Electric
Thermal (sCO2) [78] Waste Heat

Gross Electric [1000] Net Electric
Gross Electric [101] BOP
Gross Electric [40] Heating System.

:Fusion Reactions #d45
:Charged Particles #2a7fff
:Neutrons #a5a
:Pulsed Inductive DEC #07a
:Bremsstrahlung → Thermal (sCO2) #e8a030
:Thermal (sCO2) #888
:Gross Electric #6a3
:Net Electric #0a0
:Waste Heat #c33
:BOP #bcbd22
:Heating System #c80
:Heating System. #c80
```

## Figure 3: p-B11 VB DEC 60% Hybrid

Placement: end of "The Bremsstrahlung Constraint"

p_fus=2290, p_rad=1901 (83%), transport=429,
DEC gets 90% of transport=386, walls get 10%=43,
brem goes to thermal

```
// p-B11 venetian blind DEC at 60%, f_dec=0.9
// 83% of fusion power radiated as bremsstrahlung → thermal
// DEC only captures 17% charged-particle margin

Fusion Reactions [2290] Charged Particles
Heating System [40] Charged Particles

// Transport = 2290 + 40 - 1901 = 429
// 90% to DEC = 386, 10% to walls = 43
Charged Particles [1901] Bremsstrahlung → Thermal (sCO2)
Charged Particles [386] Venetian Blind DEC
Charged Particles [43] Thermal (sCO2)

Bremsstrahlung → Thermal (sCO2) [894] Gross Electric
Bremsstrahlung → Thermal (sCO2) [1007] Waste Heat

Venetian Blind DEC [232] Gross Electric
Venetian Blind DEC [154] Waste Heat

Thermal (sCO2) [20] Gross Electric
Thermal (sCO2) [23] Waste Heat

Gross Electric [1000] Net Electric
Gross Electric [106] BOP
Gross Electric [40] Heating System.

:Fusion Reactions #d45
:Charged Particles #2a7fff
:Venetian Blind DEC #07a
:Bremsstrahlung → Thermal (sCO2) #e8a030
:Thermal (sCO2) #888
:Gross Electric #6a3
:Net Electric #0a0
:Waste Heat #c33
:BOP #bcbd22
:Heating System #c80
:Heating System. #c80
```

## Figure 4: D-He3 Pulsed Inductive DEC 85% — Helion-likely operating point

Placement: in the "Pulsed Inductive (D-He3)" section.

Helion-likely operating point: D-rich mix (n_D/n_³He ≈ 3) at T ≈ 100 keV,
T not confined long enough to burn (f_T = 0), 99% inter-shot He-3 recovery,
no thermal bottoming cycle (eta_th = 0). Bosch-Hale + relativistic-bremsstrahlung
calculation (see `examples/dhe3_mix_optimization.py`) gives f_DD = 0.314,
f_brem = 0.163.

p_fus=1695, p_ash=1655, p_neutron=39, p_rad=276 (16%), transport=1419,
DEC gets 95% of transport=1348, all of brem + neutrons + charged leakage
goes to cooling as waste heat (no thermal cycle).

```
// D-He3 pulsed inductive DEC at 85%, Helion-likely operating point
// D-rich mix (f_DD = 0.314), T exhausted (f_T = 0), 99% He-3 recovery
// No thermal bottoming: bremsstrahlung, neutrons, charged-particle
// leakage, and DEC waste heat all dumped to cooling

Fusion Reactions [1655] Charged Particles
Fusion Reactions [39] Neutrons
Heating System [40] Charged Particles

// Transport = 1655 + 40 - 276 = 1419
// 95% to DEC = 1348, 5% to walls = 71
Charged Particles [276] Bremsstrahlung
Charged Particles [1348] Pulsed Inductive DEC
Charged Particles [71] Wall Losses

Pulsed Inductive DEC [1146] Gross Electric
Pulsed Inductive DEC [202] Waste Heat

Bremsstrahlung [276] Waste Heat
Wall Losses [71] Waste Heat
Neutrons [39] Waste Heat

Gross Electric [1000] Net Electric
Gross Electric [106] BOP
Gross Electric [40] Heating System.

:Fusion Reactions #d45
:Charged Particles #2a7fff
:Neutrons #a5a
:Pulsed Inductive DEC #07a
:Bremsstrahlung #e8a030
:Wall Losses #888
:Gross Electric #6a3
:Net Electric #0a0
:Waste Heat #c33
:BOP #bcbd22
:Heating System #c80
:Heating System. #c80
```
