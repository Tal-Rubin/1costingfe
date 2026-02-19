"""Layer 4: Costs — CAS10-CAS90 per-account costing.

All functions are pure (no side effects). Each takes a CostingConstants
object as first argument — no inline magic numbers.
Costs returned in millions USD (M$).

Source: pyFECONs costing/calculations/cas*.py
"""

import math

from costingfe.defaults import CostingConstants
from costingfe.layers.economics import compute_effective_crf, levelized_annual_cost
from costingfe.layers.physics import M_DEUTERIUM_KG, MEV_TO_JOULES, Q_DT
from costingfe.types import Fuel


def _total_project_time(cc, construction_time, fuel, noak):
    if noak:
        return construction_time
    return construction_time + cc.licensing_time(fuel)


# ---------------------------------------------------------------------------
# CAS Accounts
# ---------------------------------------------------------------------------


def cas10_preconstruction(cc, p_net, n_mod, fuel, noak):
    """CAS10: Pre-construction costs. Returns M$."""
    land = cc.land_intensity * p_net * math.sqrt(n_mod) * cc.land_cost / 1e6
    licensing = cc.licensing_cost(fuel)
    studies = cc.plant_studies_noak if noak else cc.plant_studies_foak
    subtotal = (
        land
        + cc.site_permits
        + licensing
        + cc.plant_permits
        + studies
        + cc.plant_reports
        + cc.other_precon
    )
    contingency = cc.contingency_rate(noak) * subtotal
    return subtotal + contingency


def cas21_buildings(cc, p_et, fuel, noak):
    """CAS21: Buildings. Scales with gross electric. Returns M$."""
    fuel_scale = 1.0 if fuel == Fuel.DT else 0.5
    tritium_buildings = (
        "site_improvements",
        "fusion_heat_island",
        "hot_cell",
        "fuel_storage",
    )
    total = 0.0
    for name, cost_per_kw in cc.building_costs_per_kw.items():
        scale = fuel_scale if name in tritium_buildings else 1.0
        total += cost_per_kw * p_et / 1000.0 * scale
    contingency = cc.contingency_rate(noak) * total
    return total + contingency


def cas23_turbine(cc, p_et, n_mod):
    """CAS23: Turbine plant equipment. Returns M$."""
    return n_mod * p_et * cc.turbine_per_mw


def cas24_electrical(cc, p_et, n_mod):
    """CAS24: Electric plant equipment. Returns M$."""
    return n_mod * p_et * cc.electric_per_mw


def cas25_misc(cc, p_et, n_mod):
    """CAS25: Miscellaneous plant equipment. Returns M$."""
    return n_mod * p_et * cc.misc_per_mw


def cas26_heat_rejection(cc, p_et, n_mod):
    """CAS26: Heat rejection. Returns M$."""
    return n_mod * p_et * cc.heat_rej_per_mw


def cas28_digital_twin(cc):
    """CAS28: Digital twin. Returns M$."""
    return cc.digital_twin


def cas29_contingency(cc, cas2x_total, noak):
    """CAS29: Contingency on direct costs. Returns M$."""
    return cc.contingency_rate(noak) * cas2x_total


def cas30_indirect(cc, cas20, p_net, construction_time):
    """CAS30: Indirect service costs. Returns M$."""
    power_scale = (p_net / cc.indirect_ref_power) ** -0.5
    field = (
        power_scale * p_net * cc.field_indirect_coeff * construction_time / 1e3
    )
    supervision = (
        power_scale
        * p_net
        * cc.construction_supervision_coeff
        * construction_time
        / 1e3
    )
    design = (
        power_scale * p_net * cc.design_services_coeff * construction_time / 1e3
    )
    return field + supervision + design


def cas40_owner(cas20):
    """CAS40: Owner's costs (~5% of direct). Returns M$."""
    return 0.05 * cas20


def cas50_supplementary(cc, cas23_to_28_total, p_net, noak):
    """CAS50: Supplementary costs. Returns M$."""
    spare_parts = cc.spare_parts_frac * cas23_to_28_total
    fuel_load = (p_net / 1000.0) * 10.0  # rough scaling
    subtotal = (
        cc.shipping
        + spare_parts
        + cc.taxes
        + cc.insurance_cost
        + fuel_load
        + cc.decommissioning
    )
    contingency = cc.contingency_rate(noak) * subtotal
    return subtotal + contingency


def cas60_idc(cc, cas20, p_net, construction_time, fuel, noak):
    """CAS60: Interest during construction. Returns M$."""
    t_project = _total_project_time(cc, construction_time, fuel, noak)
    return p_net * cc.idc_coeff * t_project / 1e3


def cas70_om(cc, p_net, inflation_rate, construction_time, fuel, noak):
    """CAS70: Annualized O&M costs (today's $ inflated to operation start). Returns M$."""
    annual_om = cc.om_cost_per_mw_yr * p_net * 1000 / 1e6  # M$
    t_project = _total_project_time(cc, construction_time, fuel, noak)
    return levelized_annual_cost(annual_om, inflation_rate, t_project)


def cas80_fuel(cc, p_fus, n_mod, availability, inflation_rate,
               construction_time, fuel, noak):
    """CAS80: Annualized fuel cost. Architecture-agnostic — depends on
    fusion power and fuel type, not confinement concept. Returns M$."""
    c_f = (
        n_mod
        * p_fus
        * 1e6
        * 3600
        * 8760
        * cc.u_deuterium
        * M_DEUTERIUM_KG
        * availability
        / (Q_DT * MEV_TO_JOULES)
    )
    annual_fuel_musd = c_f / 1e6
    t_project = _total_project_time(cc, construction_time, fuel, noak)
    return levelized_annual_cost(annual_fuel_musd, inflation_rate, t_project)


def cas90_financial(cc, total_capital, interest_rate, plant_lifetime,
                    construction_time, fuel, noak):
    """CAS90: Annualized financial (capital) costs. Returns M$."""
    t_project = _total_project_time(cc, construction_time, fuel, noak)
    eff_crf = compute_effective_crf(interest_rate, plant_lifetime, t_project)
    return eff_crf * total_capital
