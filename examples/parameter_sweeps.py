"""Example: Parameter sweeps — which lever moves LCOE the most?"""

from costingfe import ConfinementConcept, CostModel, Fuel

model = CostModel(concept=ConfinementConcept.TOKAMAK, fuel=Fuel.DT)
base = model.forward(net_electric_mw=1000.0, availability=0.85, lifetime_yr=30)
base_lcoe = base.costs.lcoe

# ── Availability sweep (0.70 → 0.95) ─────────────────────────────
avail_vals = [0.70 + i * 0.025 for i in range(11)]
avail_lcoes = model.batch_lcoe(
    param_sets={"availability": avail_vals},
    params=base.params,
)

print("Availability Sweep — DT Tokamak, 1 GWe")
print(f"{'Availability':>14} {'LCOE':>10} {'Δ LCOE':>10}")
print(f"{'':>14} {'$/MWh':>10} {'$/MWh':>10}")
print("-" * 36)
for a, lc in zip(avail_vals, avail_lcoes):
    print(f"{a:>14.3f} {lc:>10.1f} {lc - base_lcoe:>+10.1f}")

# ── Thermal-efficiency sweep (0.30 → 0.50) ───────────────────────
eta_vals = [0.30 + i * 0.02 for i in range(11)]
eta_lcoes = model.batch_lcoe(
    param_sets={"eta_th": eta_vals},
    params=base.params,
)

print("\nThermal Efficiency Sweep — DT Tokamak, 1 GWe")
print(f"{'eta_th':>14} {'LCOE':>10} {'Δ LCOE':>10}")
print(f"{'':>14} {'$/MWh':>10} {'$/MWh':>10}")
print("-" * 36)
for e, lc in zip(eta_vals, eta_lcoes):
    print(f"{e:>14.2f} {lc:>10.1f} {lc - base_lcoe:>+10.1f}")

# ── Compare leverage via sensitivity elasticity ───────────────────
sens = model.sensitivity(base.params)
eng = sens["engineering"]
e_avail = eng.get("availability", 0.0)
e_eta = eng.get("eta_th", 0.0)
print("\nSensitivity Elasticity (% LCOE / % param)")
print(f"  availability : {e_avail:+.4f}")
print(f"  eta_th       : {e_eta:+.4f}")
winner = "availability" if abs(e_avail) > abs(e_eta) else "eta_th"
print(f"\n  → {winner} has more leverage on LCOE.")
