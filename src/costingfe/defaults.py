"""Load and manage default parameters from YAML files."""

import yaml
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Dict

_DATA_DIR = Path(__file__).parent / "data" / "defaults"


@dataclass(frozen=True)
class CostingConstants:
    """All costing coefficients. Immutable — use .replace() for overrides."""

    # CAS10
    site_permits: float = 3.0
    plant_studies_foak: float = 20.0
    plant_studies_noak: float = 4.0
    plant_permits: float = 2.0
    plant_reports: float = 1.0
    other_precon: float = 1.0
    land_intensity: float = 0.001
    land_cost: float = 10000.0
    licensing_cost_dt: float = 5.0
    licensing_cost_dd: float = 3.0
    licensing_cost_dhe3: float = 1.0
    licensing_cost_pb11: float = 0.1
    licensing_time_dt: float = 5.0
    licensing_time_dd: float = 3.0
    licensing_time_dhe3: float = 2.0
    licensing_time_pb11: float = 1.0

    # CAS21
    building_costs_per_kw: Dict[str, float] = None  # loaded from YAML

    # CAS23-26
    turbine_per_mw: float = 0.19764
    electric_per_mw: float = 0.08418
    misc_per_mw: float = 0.05124
    heat_rej_per_mw: float = 0.03416

    # CAS28
    digital_twin: float = 5.0

    # CAS29
    contingency_rate_foak: float = 0.10
    contingency_rate_noak: float = 0.0

    # CAS30
    field_indirect_coeff: float = 0.4
    construction_supervision_coeff: float = 0.4
    design_services_coeff: float = 0.4
    indirect_ref_power: float = 1000.0

    # CAS50
    shipping: float = 1.0
    spare_parts_frac: float = 0.01
    taxes: float = 0.5
    insurance_cost: float = 0.5
    decommissioning: float = 5.0

    # CAS60
    idc_coeff: float = 0.05

    # CAS70
    om_cost_per_mw_yr: float = 60.0

    # CAS80 — STARFIRE (1980) inflation-adjusted via GDP IPD. Range: $1,500-3,500/kg.
    u_deuterium: float = 2175.0  # $/kg

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    def licensing_cost(self, fuel):
        from costingfe.types import Fuel

        return {
            Fuel.DT: self.licensing_cost_dt,
            Fuel.DD: self.licensing_cost_dd,
            Fuel.DHE3: self.licensing_cost_dhe3,
            Fuel.PB11: self.licensing_cost_pb11,
        }.get(fuel, self.licensing_cost_dt)

    def licensing_time(self, fuel):
        from costingfe.types import Fuel

        return {
            Fuel.DT: self.licensing_time_dt,
            Fuel.DD: self.licensing_time_dd,
            Fuel.DHE3: self.licensing_time_dhe3,
            Fuel.PB11: self.licensing_time_pb11,
        }.get(fuel, self.licensing_time_dt)

    def contingency_rate(self, noak):
        return self.contingency_rate_noak if noak else self.contingency_rate_foak


def load_costing_constants(path: Path = None) -> CostingConstants:
    """Load costing constants from YAML, falling back to dataclass defaults."""
    if path is None:
        path = _DATA_DIR / "costing_constants.yaml"
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)
        valid_fields = {f.name for f in fields(CostingConstants)}
        return CostingConstants(**{k: v for k, v in data.items() if k in valid_fields})
    return CostingConstants()


def load_engineering_defaults(concept_fuel: str) -> dict:
    """Load engineering defaults for a concept (e.g., 'mfe_tokamak')."""
    path = _DATA_DIR / f"{concept_fuel}.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}
