"""Input validation for the costing model.

Pydantic-based CostingInput with three validation tiers:
- Tier 1: Field-level constraints (pydantic Field)
- Tier 2: Family-aware required engineering parameters
- Tier 3: Cross-field physics checks
"""

import warnings
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from costingfe.types import (
    CONCEPT_TO_FAMILY,
    ConfinementConcept,
    ConfinementFamily,
    Fuel,
)


class CostingInput(BaseModel):
    """Validated input for the costing model.

    Required fields: concept, fuel, net_electric_mw.
    Customer parameters have defaults.
    Engineering parameters default to None (filled from YAML templates).
    """

    # --- Required (no defaults) ---
    concept: ConfinementConcept
    fuel: Fuel
    net_electric_mw: float = Field(gt=0)

    # --- Customer parameters (with defaults) ---
    availability: float = Field(default=0.85, gt=0, le=1)
    lifetime_yr: float = Field(default=40.0, gt=0)
    n_mod: int = Field(default=1, ge=1, strict=True)
    construction_time_yr: float = Field(default=6.0, gt=0)
    interest_rate: float = Field(default=0.07, gt=0)
    inflation_rate: float = 0.02
    noak: bool = True
    cost_overrides: dict[str, float] = Field(default_factory=dict)
    costing_overrides: dict[str, float] = Field(default_factory=dict)

    # --- Engineering parameters (None = use YAML template) ---
    # Common (all families)
    mn: Optional[float] = None
    eta_th: Optional[float] = None
    eta_p: Optional[float] = None
    f_sub: Optional[float] = None
    p_pump: Optional[float] = None
    p_trit: Optional[float] = None
    p_house: Optional[float] = None
    p_cryo: Optional[float] = None
    blanket_t: Optional[float] = None
    ht_shield_t: Optional[float] = None
    structure_t: Optional[float] = None
    vessel_t: Optional[float] = None
    plasma_t: Optional[float] = None

    # MFE only
    p_input: Optional[float] = None
    eta_pin: Optional[float] = None
    eta_de: Optional[float] = None
    f_dec: Optional[float] = None
    p_coils: Optional[float] = None
    p_cool: Optional[float] = None
    axis_t: Optional[float] = None
    elon: Optional[float] = None

    # IFE only
    p_implosion: Optional[float] = None
    p_ignition: Optional[float] = None
    eta_pin1: Optional[float] = None
    eta_pin2: Optional[float] = None
    p_target: Optional[float] = None  # shared with MIF

    # MIF only
    p_driver: Optional[float] = None
    # eta_pin: already declared above (shared MFE/MIF)
    # p_target: already declared above (shared IFE/MIF)
    # p_coils: already declared above (shared MFE/MIF)

    # Plasma parameters (MFE radiation calculation)
    n_e: Optional[float] = None
    T_e: Optional[float] = None
    Z_eff: Optional[float] = None
    plasma_volume: Optional[float] = None
    B: Optional[float] = None

    # --- Tier 2: family-required parameter lists ---
    _COMMON_REQUIRED = [
        "mn", "eta_th", "eta_p", "f_sub",
        "p_pump", "p_trit", "p_house", "p_cryo",
        "blanket_t", "ht_shield_t", "structure_t", "vessel_t", "plasma_t",
    ]
    _MFE_REQUIRED = [
        "p_input", "eta_pin", "eta_de", "f_dec",
        "p_coils", "p_cool", "axis_t", "elon",
    ]
    _IFE_REQUIRED = [
        "p_implosion", "p_ignition", "eta_pin1", "eta_pin2", "p_target",
    ]
    _MIF_REQUIRED = [
        "p_driver", "eta_pin", "p_target", "p_coils",
    ]

    @model_validator(mode="after")
    def check_family_required_params(self):
        """Tier 2: If any engineering param is set, all family-required params must be present."""
        family = CONCEPT_TO_FAMILY[self.concept]

        all_eng = (
            self._COMMON_REQUIRED
            + self._MFE_REQUIRED + self._IFE_REQUIRED + self._MIF_REQUIRED
        )
        any_set = any(getattr(self, k) is not None for k in all_eng)
        if not any_set:
            return self

        family_required = {
            ConfinementFamily.MFE: self._MFE_REQUIRED,
            ConfinementFamily.IFE: self._IFE_REQUIRED,
            ConfinementFamily.MIF: self._MIF_REQUIRED,
        }
        required = self._COMMON_REQUIRED + family_required.get(family, [])

        missing = [k for k in required if getattr(self, k) is None]
        if missing:
            raise ValueError(
                f"Missing required engineering parameters for "
                f"{family.value}: {', '.join(missing)}"
            )
        return self
