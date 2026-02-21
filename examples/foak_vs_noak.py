"""Example: FOAK vs NOAK — first-of-a-kind cost premium."""

from costingfe import ConfinementConcept, CostModel, Fuel
from costingfe.defaults import load_costing_constants

cc = load_costing_constants()


def run_pair(label, concept, fuel):
    """Run FOAK and NOAK side-by-side for one concept/fuel."""
    model = CostModel(concept=concept, fuel=fuel)
    foak = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        inflation_rate=0.0245,
        noak=False,
    )
    noak = model.forward(
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        inflation_rate=0.0245,
        noak=True,
    )

    lic_time = cc.licensing_time(fuel)
    cont_foak = cc.contingency_rate(noak=False)
    cont_noak = cc.contingency_rate(noak=True)

    print(f"\n{label}")
    print(f"{'':20} {'FOAK':>12} {'NOAK':>12} {'Δ':>10}")
    print(f"{'':20} {'':>12} {'':>12} {'':>10}")
    print("-" * 56)
    print(f"{'Licensing time (yr)':<20} {lic_time:>12.1f} {'—':>12} {'':>10}")
    print(f"{'Contingency rate':<20} {cont_foak:>11.0%} {cont_noak:>11.0%} {'':>10}")
    print(
        f"{'Total capital (M$)':<20} {foak.costs.total_capital:>12.0f} "
        f"{noak.costs.total_capital:>12.0f} "
        f"{foak.costs.total_capital - noak.costs.total_capital:>+10.0f}"
    )
    print(
        f"{'LCOE ($/MWh)':<20} {foak.costs.lcoe:>12.1f} "
        f"{noak.costs.lcoe:>12.1f} "
        f"{foak.costs.lcoe - noak.costs.lcoe:>+10.1f}"
    )


print("FOAK vs NOAK Comparison")
print("=" * 56)
run_pair(
    "DT Tokamak — high licensing burden",
    ConfinementConcept.TOKAMAK,
    Fuel.DT,
)
run_pair(
    "pB11 Tokamak — low licensing (aneutronic)",
    ConfinementConcept.TOKAMAK,
    Fuel.PB11,
)
