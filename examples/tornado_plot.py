"""Example: Sensitivity tornado chart — which parameters matter most?"""

from costingfe import ConfinementConcept, CostModel, Fuel

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
base = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)

sens = model.sensitivity(base.params)

# Merge engineering + financial, skip near-zero
all_params = {}
for category in ("engineering", "financial"):
    for param, elasticity in sens[category].items():
        if abs(elasticity) > 1e-4:
            all_params[param] = elasticity

# Sort by absolute elasticity (largest first)
ranked = sorted(all_params.items(), key=lambda x: abs(x[1]), reverse=True)

# ── ASCII tornado chart ──────────────────────────────────────────
BAR_WIDTH = 30
max_abs = max(abs(e) for _, e in ranked) if ranked else 1.0

print("Sensitivity Tornado — DT Tokamak, 1 GWe")
print(f"Baseline LCOE: {base.costs.lcoe:.1f} $/MWh\n")
print(
    f"{'Parameter':<24} {'Elasticity':>10}  {'← cost decrease':>{BAR_WIDTH}} "
    f"| {'cost increase →'}"
)
print("-" * (24 + 10 + 2 + BAR_WIDTH + 3 + BAR_WIDTH))

for param, elast in ranked:
    bar_len = int(abs(elast) / max_abs * BAR_WIDTH)
    if elast < 0:
        left_bar = " " * (BAR_WIDTH - bar_len) + "\u2588" * bar_len
        right_bar = ""
    else:
        left_bar = " " * BAR_WIDTH
        right_bar = "\u2588" * bar_len
    print(f"{param:<24} {elast:>+10.4f}  {left_bar} | {right_bar}")

print("\nElasticity = % change in LCOE per 1% change in parameter.")
print(f"Top lever: {ranked[0][0]} ({ranked[0][1]:+.4f})" if ranked else "")
