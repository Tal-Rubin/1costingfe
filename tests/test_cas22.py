from costingfe.defaults import load_costing_constants
from costingfe.layers.cas22 import cas22_reactor_plant_equipment
from costingfe.types import Fuel

CC = load_costing_constants()


def test_cas22_dt_has_breeding_blanket():
    """DT should include tritium breeding blanket cost."""
    result = cas22_reactor_plant_equipment(
        CC, p_net=1000.0, p_th=2500.0, p_et=1100.0, p_fus=2300.0,
        p_cryo=0.5, n_mod=1, fuel=Fuel.DT, noak=True,
    )
    assert result["C220101"] > 0  # first wall + blanket
    assert result["C220000"] > 0  # total


def test_cas22_pb11_no_breeding():
    """pB11 should have cheaper blanket (no breeding)."""
    dt = cas22_reactor_plant_equipment(
        CC, p_net=1000.0, p_th=2500.0, p_et=1100.0, p_fus=2300.0,
        p_cryo=0.5, n_mod=1, fuel=Fuel.DT, noak=True,
    )
    pb11 = cas22_reactor_plant_equipment(
        CC, p_net=1000.0, p_th=2500.0, p_et=1100.0, p_fus=2300.0,
        p_cryo=0.5, n_mod=1, fuel=Fuel.PB11, noak=True,
    )
    assert pb11["C220101"] < dt["C220101"]  # no breeding blanket


def test_cas22_isotope_separation_fuel_dependent():
    """Isotope separation should vary by fuel type."""
    dt = cas22_reactor_plant_equipment(
        CC, p_net=1000.0, p_th=2500.0, p_et=1100.0, p_fus=2300.0,
        p_cryo=0.5, n_mod=1, fuel=Fuel.DT, noak=True,
    )
    pb11 = cas22_reactor_plant_equipment(
        CC, p_net=1000.0, p_th=2500.0, p_et=1100.0, p_fus=2300.0,
        p_cryo=0.5, n_mod=1, fuel=Fuel.PB11, noak=True,
    )
    # Both should have isotope separation but different amounts
    assert dt["C220112"] > 0
    assert pb11["C220112"] > 0
    assert dt["C220112"] != pb11["C220112"]


def test_cas22_fuel_handling_tritium_containment():
    """DT should have much higher fuel handling cost (tritium containment)."""
    dt = cas22_reactor_plant_equipment(
        CC, p_net=1000.0, p_th=2500.0, p_et=1100.0, p_fus=2300.0,
        p_cryo=0.5, n_mod=1, fuel=Fuel.DT, noak=True,
    )
    pb11 = cas22_reactor_plant_equipment(
        CC, p_net=1000.0, p_th=2500.0, p_et=1100.0, p_fus=2300.0,
        p_cryo=0.5, n_mod=1, fuel=Fuel.PB11, noak=True,
    )
    assert dt["C220500"] > pb11["C220500"]


def test_cas22_scales_with_n_mod():
    """Total CAS22 should scale with number of modules."""
    single = cas22_reactor_plant_equipment(
        CC, p_net=1000.0, p_th=2500.0, p_et=1100.0, p_fus=2300.0,
        p_cryo=0.5, n_mod=1, fuel=Fuel.DT, noak=True,
    )
    double = cas22_reactor_plant_equipment(
        CC, p_net=1000.0, p_th=2500.0, p_et=1100.0, p_fus=2300.0,
        p_cryo=0.5, n_mod=2, fuel=Fuel.DT, noak=True,
    )
    assert double["C220000"] > single["C220000"]


def test_cas22_all_subaccounts_present():
    """Result should contain all expected sub-account keys."""
    result = cas22_reactor_plant_equipment(
        CC, p_net=1000.0, p_th=2500.0, p_et=1100.0, p_fus=2300.0,
        p_cryo=0.5, n_mod=1, fuel=Fuel.DT, noak=True,
    )
    expected_keys = [
        "C220101", "C220102", "C220103", "C220104", "C220105",
        "C220106", "C220107", "C220108", "C220109", "C220111",
        "C220112", "C220119", "C220200", "C220300", "C220400",
        "C220500", "C220600", "C220700", "C220000",
    ]
    for key in expected_keys:
        assert key in result, f"Missing sub-account {key}"
        assert result[key] >= 0, f"Sub-account {key} is negative"
