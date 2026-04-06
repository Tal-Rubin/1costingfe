"""Tests for pulsed thermal power balance (forward + inverse)."""

from costingfe.layers.physics import (
    pulsed_thermal_forward,
    pulsed_thermal_inverse,
)
from costingfe.types import Fuel

THERMAL_PARAMS = dict(
    p_fus=2500.0,
    fuel=Fuel.DT,
    e_driver_mj=100.0,
    f_rep=1.0,
    mn=1.1,
    eta_th=0.40,
    eta_pin=0.15,
    f_rad=0.10,
    f_sub=0.03,
    p_pump=1.0,
    p_trit=10.0,
    p_house=4.0,
    p_cryo=0.5,
    p_target=1.0,
    p_coils=0.0,
)


def test_thermal_forward_positive_net():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    assert pt.p_net > 0


def test_thermal_forward_no_dec():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    assert pt.p_dee == 0.0


def test_thermal_forward_pulsed_fields():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    assert pt.e_driver_mj == 100.0
    assert pt.f_rep == 1.0
    assert pt.e_stored_mj > pt.e_driver_mj
    assert pt.f_ch > 0


def test_thermal_forward_driver_thermalizes():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    assert pt.p_th > pt.p_fus * 1.1  # p_th includes driver power


def test_thermal_forward_no_pump_without_thermal():
    params = dict(THERMAL_PARAMS, eta_th=0.0)
    pt = pulsed_thermal_forward(**params)
    assert pt.p_et == 0.0


def test_thermal_inverse_roundtrip():
    pt = pulsed_thermal_forward(**THERMAL_PARAMS)
    inv_params = {k: v for k, v in THERMAL_PARAMS.items() if k not in ("p_fus", "fuel")}
    p_fus_recovered = pulsed_thermal_inverse(
        p_net_target=pt.p_net, fuel=Fuel.DT, **inv_params
    )
    assert abs(p_fus_recovered - 2500.0) < 0.5
