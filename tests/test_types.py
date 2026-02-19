from costingfe.types import (
    ConfinementFamily,
    ConfinementConcept,
    Fuel,
    CONCEPT_TO_FAMILY,
)


def test_concept_to_family_mapping():
    assert CONCEPT_TO_FAMILY[ConfinementConcept.TOKAMAK] == ConfinementFamily.MFE
    assert CONCEPT_TO_FAMILY[ConfinementConcept.LASER_IFE] == ConfinementFamily.IFE
    assert CONCEPT_TO_FAMILY[ConfinementConcept.MAG_TARGET] == ConfinementFamily.MIF


def test_all_concepts_have_family():
    for concept in ConfinementConcept:
        assert concept in CONCEPT_TO_FAMILY, f"{concept} missing from CONCEPT_TO_FAMILY"


def test_fuel_enum():
    assert len(Fuel) == 4
    assert Fuel.DT.value == "dt"
