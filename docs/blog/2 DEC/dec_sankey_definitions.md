# Sankey Diagram Definitions for "Direct Energy Conversion and the Cost Floor"

Generated using [SankeyMatic](https://sankeymatic.com/build/). Paste each block into the editor.

All energy values in MW. All cases are 1 GWe net output, baseline conditions
(85% availability, 7% WACC, 30-year life, 6-year construction).

Power balance from `1costingfe` model (mirror concept, f_rad_fus=0.87 for p-B11,
f_rad_fus=0.25 for D-He3).

"Heating System." (with period) is a SankeyMatic trick to create a separate node
from the left-side "Heating System". After export, edit the SVG to reroute the
flow back to the left node, remove the period, and move the label into the loop.

## Figure 1: p-B11 Thermal Cycle

Placement: end of "What Direct Energy Conversion Does"

p_fus=2397, p_rad=2086 (87%), transport=352, p_th=2438, p_the=1146

```
// p-B11 thermal only (f_dec=0), 1 GWe net output
// 87% of fusion power radiated as bremsstrahlung
// Heating: 80 MW wall-plug, 40 MW delivered to plasma

Fusion Reactions [2397] Charged Particles
Heating System [40] Charged Particles

// Bremsstrahlung dominates: 2086 MW radiated, 352 MW transport
Charged Particles [2086] Bremsstrahlung → Thermal (sCO2)
Charged Particles [352] Transport → Thermal (sCO2)

Bremsstrahlung → Thermal (sCO2) [980] Gross Electric
Bremsstrahlung → Thermal (sCO2) [1106] Waste Heat

Transport → Thermal (sCO2) [166] Gross Electric
Transport → Thermal (sCO2) [186] Waste Heat

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

## Figure 2: D-He3 Pulsed Inductive DEC 85%

Placement: end of "Pulsed Inductive DEC"

p_fus=1533, p_ash=1462, p_neutron=71, p_rad=383 (25%), transport=1119,
DEC gets 95% of transport=1063, thermal gets brem+neutrons+5% transport

```
// D-He3 pulsed inductive DEC at 85%, f_dec=0.95
// 25% of fusion power radiated as bremsstrahlung (consensus clean-plasma value)
// 95% of charged transport to DEC, 5% + brem + neutrons to thermal

Fusion Reactions [1462] Charged Particles
Fusion Reactions [71] Neutrons
Heating System [40] Charged Particles

// Transport = 1462 + 40 - 383 = 1119
// 95% to DEC = 1063, 5% to walls = 56
Charged Particles [383] Bremsstrahlung → Thermal (sCO2)
Charged Particles [1063] Pulsed Inductive DEC
Charged Particles [56] Thermal (sCO2)
Neutrons [78] Thermal (sCO2)

Pulsed Inductive DEC [904] Gross Electric
Pulsed Inductive DEC [159] Waste Heat

Bremsstrahlung → Thermal (sCO2) [180] Gross Electric
Bremsstrahlung → Thermal (sCO2) [203] Waste Heat

Thermal (sCO2) [63] Gross Electric
Thermal (sCO2) [71] Waste Heat

Gross Electric [1000] Net Electric
Gross Electric [106] BOP
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

p_fus=2313, p_rad=2012 (87%), transport=341,
DEC gets 90% of transport=307, walls get 10%=34,
brem goes to thermal

```
// p-B11 venetian blind DEC at 60%, f_dec=0.9
// 87% of fusion power radiated as bremsstrahlung → thermal
// DEC only captures 13% charged-particle margin

Fusion Reactions [2313] Charged Particles
Heating System [40] Charged Particles

// Transport = 2313 + 40 - 2012 = 341
// 90% to DEC = 307, 10% to walls = 34
Charged Particles [2012] Bremsstrahlung → Thermal (sCO2)
Charged Particles [307] Venetian Blind DEC
Charged Particles [34] Thermal (sCO2)

Bremsstrahlung → Thermal (sCO2) [946] Gross Electric
Bremsstrahlung → Thermal (sCO2) [1066] Waste Heat

Venetian Blind DEC [184] Gross Electric
Venetian Blind DEC [123] Waste Heat

Thermal (sCO2) [16] Gross Electric
Thermal (sCO2) [18] Waste Heat

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
