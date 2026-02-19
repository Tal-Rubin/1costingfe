from costingfe.defaults import load_costing_constants
from costingfe.layers.costs import (
    cas10_preconstruction,
    cas21_buildings,
    cas23_turbine,
    cas24_electrical,
    cas25_misc,
    cas26_heat_rejection,
    cas28_digital_twin,
    cas29_contingency,
    cas30_indirect,
    cas40_owner,
    cas50_supplementary,
    cas60_idc,
    cas70_om,
    cas80_fuel,
    cas90_financial,
)
from costingfe.types import Fuel

CC = load_costing_constants()


def test_cas10_dt_licensing():
    """DT licensing should be $5M."""
    cost = cas10_preconstruction(CC, p_net=1000.0, n_mod=1, fuel=Fuel.DT, noak=True)
    assert cost > 0


def test_cas10_pb11_cheaper_licensing():
    """pB11 licensing should be cheaper than DT."""
    cost_dt = cas10_preconstruction(
        CC, p_net=1000.0, n_mod=1, fuel=Fuel.DT, noak=True
    )
    cost_pb11 = cas10_preconstruction(
        CC, p_net=1000.0, n_mod=1, fuel=Fuel.PB11, noak=True
    )
    assert cost_pb11 < cost_dt


def test_cas21_scales_with_power():
    """Building costs should scale with gross electric power."""
    cost_low = cas21_buildings(CC, p_et=500.0, fuel=Fuel.DT, noak=True)
    cost_high = cas21_buildings(CC, p_et=1000.0, fuel=Fuel.DT, noak=True)
    assert cost_high > cost_low


def test_cas23_to_26_scale_with_power():
    """BOP equipment scales with gross electric power."""
    c23 = cas23_turbine(CC, p_et=1000.0, n_mod=1)
    c24 = cas24_electrical(CC, p_et=1000.0, n_mod=1)
    c25 = cas25_misc(CC, p_et=1000.0, n_mod=1)
    c26 = cas26_heat_rejection(CC, p_et=1000.0, n_mod=1)
    for c in [c23, c24, c25, c26]:
        assert c > 0


def test_cas90_annualizes_capital():
    """CAS90 should be CRF * total capital."""
    c90 = cas90_financial(
        CC,
        total_capital=5000.0,
        interest_rate=0.07,
        plant_lifetime=30,
        construction_time=6,
        fuel=Fuel.DT,
        noak=True,
    )
    assert c90 > 0
    assert c90 < 5000  # annualized should be less than total


def test_lcoe_end_to_end_sanity():
    """Full cost stack should produce LCOE in reasonable range."""
    from costingfe.layers.economics import compute_lcoe

    c90 = cas90_financial(CC, 5000.0, 0.07, 30, 6, Fuel.DT, True)
    c70 = cas70_om(
        CC, p_net=1000.0, inflation_rate=0.0245,
        construction_time=6, fuel=Fuel.DT, noak=True,
    )
    c80 = cas80_fuel(
        CC, p_fus=2600.0, n_mod=1, availability=0.85,
        inflation_rate=0.0245, construction_time=6, fuel=Fuel.DT, noak=True,
    )
    lcoe = compute_lcoe(c90, c70, c80, p_net=1000.0, n_mod=1, availability=0.85)
    assert 10 < lcoe < 500, f"LCOE {lcoe} $/MWh out of expected range"
