"""Example: Cost overrides — inject vendor quotes into the model."""

from costingfe import ConfinementConcept, CostModel, Fuel

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)

# ── Baseline ──────────────────────────────────────────────────────
base = model.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
    inflation_rate=0.0245,
)

# ── With vendor quotes ────────────────────────────────────────────
# Suppose a coil vendor quotes 20% below our default C220103,
# and a site survey pins CAS21 (buildings) at 250 M$.
coil_quote = base.cas22_detail["C220103"] * 0.80
building_quote = 250.0

override = model.forward(
    net_electric_mw=1000.0,
    availability=0.85,
    lifetime_yr=30,
    inflation_rate=0.0245,
    cost_overrides={"C220103": coil_quote, "CAS21": building_quote},
)

# ── Results ───────────────────────────────────────────────────────
print("Cost Overrides — DT Tokamak, 1 GWe")
print(f"{'':24} {'Baseline':>12} {'Overridden':>12} {'Δ':>10}")
print(f"{'':24} {'M$':>12} {'M$':>12} {'M$':>10}")
print("-" * 60)
print(
    f"{'C220103 Coils':<24} {base.cas22_detail['C220103']:>12.1f} "
    f"{coil_quote:>12.1f} "
    f"{coil_quote - base.cas22_detail['C220103']:>+10.1f}"
)
print(
    f"{'CAS21 Buildings':<24} {base.costs.cas21:>12.1f} "
    f"{building_quote:>12.1f} "
    f"{building_quote - base.costs.cas21:>+10.1f}"
)
print("-" * 60)
print(
    f"{'Total Capital':<24} {base.costs.total_capital:>12.0f} "
    f"{override.costs.total_capital:>12.0f} "
    f"{override.costs.total_capital - base.costs.total_capital:>+10.0f}"
)
print(
    f"{'LCOE ($/MWh)':<24} {base.costs.lcoe:>12.1f} "
    f"{override.costs.lcoe:>12.1f} "
    f"{override.costs.lcoe - base.costs.lcoe:>+10.1f}"
)
print(f"\nOverridden accounts: {', '.join(override.overridden)}")
