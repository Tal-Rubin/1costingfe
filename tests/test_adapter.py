from costingfe.adapter import FusionTeaInput, run_costing


def test_adapter_tokamak_dt():
    """Adapter should produce valid output for a tokamak DT plant."""
    inp = FusionTeaInput(
        concept="tokamak",
        fuel="dt",
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
    )
    out = run_costing(inp)
    assert out.lcoe > 0
    assert out.overnight_cost > 0
    assert out.total_capital > 0
    assert "CAS22" in out.costs
    assert out.costs["CAS22"] > 0
    assert "p_fus" in out.power_table
    assert out.power_table["p_fus"] > 0
    assert "eta_th" in out.sensitivity["engineering"]


def test_adapter_ife_pb11():
    """Adapter should work for IFE pB11."""
    inp = FusionTeaInput(
        concept="laser_ife",
        fuel="pb11",
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
    )
    out = run_costing(inp)
    assert out.lcoe > 0
    assert out.power_table["q_eng"] > 1.0


def test_adapter_with_overrides():
    """Adapter should accept engineering overrides."""
    inp = FusionTeaInput(
        concept="tokamak",
        fuel="dt",
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
        overrides={"eta_th": 0.50},
    )
    out = run_costing(inp)

    inp_base = FusionTeaInput(
        concept="tokamak",
        fuel="dt",
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
    )
    out_base = run_costing(inp_base)

    # Higher eta_th should reduce LCOE
    assert out.lcoe < out_base.lcoe


def test_adapter_cas_codes_complete():
    """Output should contain all CAS codes."""
    inp = FusionTeaInput(
        concept="tokamak",
        fuel="dt",
        net_electric_mw=1000.0,
        availability=0.85,
        lifetime_yr=30,
    )
    out = run_costing(inp)
    expected = [
        "CAS10", "CAS20", "CAS21", "CAS22", "CAS23", "CAS24",
        "CAS25", "CAS26", "CAS27", "CAS28", "CAS29", "CAS30",
        "CAS40", "CAS50", "CAS60", "CAS70", "CAS80", "CAS90",
    ]
    for code in expected:
        assert code in out.costs, f"Missing {code}"
