"""Layer 2: Physics — fuel physics, power balance (forward + inverse)."""

import jax.numpy as jnp
from scipy import constants as sc

from costingfe.types import Fuel, ImpurityMix, PowerTable, WallMaterial

# ---------------------------------------------------------------------------
# Fundamental constants from scipy (CODATA)
# ---------------------------------------------------------------------------
MEV_TO_JOULES = sc.eV * 1e6  # 1 MeV in J
M_DEUTERIUM_KG = sc.physical_constants["deuteron mass"][0]  # kg
M_PROTON_KG = sc.physical_constants["proton mass"][0]  # kg
M_HE3_KG = sc.physical_constants["helion mass"][0]  # kg (helion = He-3 nucleus)
M_LI6_KG = 6.015122795 * sc.atomic_mass  # kg (not in scipy CODATA)
M_B11_KG = 11.0093054 * sc.atomic_mass  # kg (not in scipy CODATA)

# DD burn fraction defaults (T=50 keV, tau_p=5 s)
DD_F_T_DEFAULT = 0.969  # DD tritium burn fraction
DD_F_HE3_DEFAULT = 0.689  # DD He-3 burn fraction

# ---------------------------------------------------------------------------
# Fusion Q-values and product energies (MeV) — nuclear reaction data
# Source: pyFECONs fuel_physics.py
# ---------------------------------------------------------------------------
# DT: D + T -> He4(3.52 MeV) + n(14.06 MeV), Q = 17.58 MeV
E_ALPHA_DT = 3.52
Q_DT = 17.58
E_N_DT = 14.06

# DD branch 1: D + D -> T(1.01 MeV) + p(3.02 MeV), Q = 4.03 MeV
E_T_DD = 1.01
E_P_DD = 3.02
Q_DD_PT = 4.03

# DD branch 2: D + D -> He3(0.82 MeV) + n(2.45 MeV), Q = 3.27 MeV
E_HE3_DD = 0.82
E_N_DD = 2.45
Q_DD_NHE3 = 3.27

# DHe3: D + He3 -> He4(3.6 MeV) + p(14.7 MeV), Q = 18.35 MeV
Q_DHE3 = 18.35

# PB11: p + B11 -> 3 He4, Q = 8.68 MeV
Q_PB11 = 8.68

# PB11 side reactions (neutron-producing)
# 11B(alpha,n)14N: Q = +0.158 MeV, dominant (~1e-3 per alpha)
Q_ALPHA_N_PB11 = 0.158
# 11B(p,n)11C: Q = -2.765 MeV, threshold 3.02 MeV (~1e-5 per proton)
Q_P_N_PB11 = -2.765

# DD primary per-event averages (50/50 branches)
_E_CHARGED_PRIMARY_DD = 0.5 * (E_T_DD + E_P_DD) + 0.5 * E_HE3_DD  # ~2.425
_E_NEUTRON_PRIMARY_DD = 0.5 * E_N_DD  # ~1.225
_E_TOTAL_PRIMARY_DD = 0.5 * Q_DD_PT + 0.5 * Q_DD_NHE3  # ~3.65


def ash_neutron_split(
    p_fus: float,
    fuel: Fuel,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
) -> tuple[float, float]:
    """Compute charged-particle (ash) and neutron power from fusion power.

    Returns (p_ash, p_neutron) in MW. All paths are JAX-differentiable.

    Source: pyFECONs fuel_physics.py:compute_ash_neutron_split
    """
    if fuel == Fuel.DT:
        ash_frac = E_ALPHA_DT / Q_DT
    elif fuel == Fuel.DD:
        E_charged = (
            _E_CHARGED_PRIMARY_DD + 0.5 * dd_f_T * E_ALPHA_DT + 0.5 * dd_f_He3 * Q_DHE3
        )
        E_total = _E_TOTAL_PRIMARY_DD + 0.5 * dd_f_T * Q_DT + 0.5 * dd_f_He3 * Q_DHE3
        ash_frac = E_charged / E_total
    elif fuel == Fuel.DHE3:
        E_n_dd = _E_NEUTRON_PRIMARY_DD + 0.5 * dhe3_f_T * E_N_DT
        E_c_dd = _E_CHARGED_PRIMARY_DD + 0.5 * dhe3_f_T * E_ALPHA_DT
        ash_frac = (1 - dhe3_dd_frac) + dhe3_dd_frac * E_c_dd / (E_n_dd + E_c_dd)
    elif fuel == Fuel.PB11:
        # Side reaction 1: 11B(alpha,n)14N
        # Each p+11B produces 3 alphas (~2.89 MeV each). Each alpha has
        # probability pb11_f_alpha_n of undergoing 11B(alpha,n)14N.
        n_alpha_n = 3.0 * pb11_f_alpha_n  # alpha-n reactions per primary event
        E_alpha = Q_PB11 / 3.0  # average alpha energy (~2.89 MeV)
        # CM kinematics: neutron energy in lab frame
        # E_CM = E_alpha * m_B11/(m_alpha + m_B11) = E_alpha * 11/15
        # Products share E_CM + Q among n(1/15) and 14N(14/15)
        E_CM_products = E_alpha * 11.0 / 15.0 + Q_ALPHA_N_PB11
        E_n_alpha = E_CM_products * 14.0 / 15.0 + E_alpha * 4.0 / 225.0  # ~2.18 MeV

        # Side reaction 2: 11B(p,n)11C
        # Endothermic (Q = -2.765 MeV, threshold 3.02 MeV). The primary
        # proton has ~zero kinetic energy in the CM, so this reaction competes
        # with p+11B only at the high-energy tail. pb11_f_p_n is the fraction
        # of primary p+11B events that instead undergo (p,n).
        # CM kinematics: E_CM = E_p * 11/12; products share E_CM + Q
        # At threshold (~3.02 MeV), E_n ~ (E_CM + Q) * 11/12 ~ 0.17 MeV
        E_n_pn = 0.17  # MeV, neutron energy near threshold

        # Total energy and neutron energy per primary event
        E_total = Q_PB11 + n_alpha_n * Q_ALPHA_N_PB11 + pb11_f_p_n * Q_P_N_PB11
        E_neutron = n_alpha_n * E_n_alpha + pb11_f_p_n * E_n_pn
        ash_frac = (E_total - E_neutron) / E_total
    else:
        raise ValueError(f"Unknown fuel type: {fuel}")

    p_ash = p_fus * ash_frac
    p_neutron = p_fus * (1.0 - ash_frac)
    return p_ash, p_neutron


# ---------------------------------------------------------------------------
# Sputtering yield model (Bohdansky/Eckstein physical sputtering)
# ---------------------------------------------------------------------------
# Fit params: (E_th_eV, Q_factor, M_target_amu)
# Y(E) = Q * S_n(E/E_TF) * (1 - (E_th/E)^(2/3)) * (1 - E_th/E)^2
# where S_n is the nuclear stopping cross-section (Thomas-Fermi)
# Source: Eckstein 2007, Bohdansky 1984
_SPUT_PARAMS: dict[str, tuple[float, float, float]] = {
    "W": (220.0, 0.042, 183.84),
    "C": (28.0, 0.060, 12.011),
    "Be": (15.0, 0.100, 9.012),
    "Mo": (100.0, 0.050, 95.95),
    "SiC": (40.0, 0.055, 20.04),  # Si-like effective values
    "Li": (10.0, 0.020, 6.941),  # Low yield from flowing surface renewal
}

# Default SOL screening factors by atomic number category
_F_SCREEN_DEFAULT = {
    "W": 0.01,
    "Mo": 0.01,  # high-Z: strong screening
    "C": 0.10,
    "Be": 0.10,
    "SiC": 0.10,  # low-Z: weaker screening
    "Li": 0.005,  # liquid Li: flowing surface, very low penetration
}


def compute_sputtering_yield(T_edge_keV: float, wall_material: WallMaterial) -> float:
    """Physical sputtering yield Y (atoms/ion) at incident energy ~3*T_edge.

    Uses simplified Bohdansky formula. T_edge_keV is the edge ion temperature.
    Incident ion energy is approximated as 3*T_edge (sheath-accelerated D ions).
    """
    symbol = wall_material.value
    E_th, Q, M_t = _SPUT_PARAMS[symbol]
    E_ion = 3.0 * T_edge_keV * 1e3  # keV -> eV

    # Below threshold: zero yield
    ratio = E_th / jnp.maximum(E_ion, 1.0)  # avoid division by zero
    below_threshold = E_ion <= E_th

    # Simplified Bohdansky: Y = Q * (1 - (E_th/E)^(2/3)) * (1 - E_th/E)^2
    Y = Q * (1.0 - ratio ** (2.0 / 3.0)) * (1.0 - ratio) ** 2
    Y = jnp.where(below_threshold, 0.0, Y)
    return Y


def compute_impurity_fraction(
    wall_material: WallMaterial,
    T_edge_keV: float,
    fw_area: float,
    plasma_volume: float,
    f_screen: float = 0.01,
    tau_ratio: float = 3.0,
) -> tuple[str, float]:
    """Steady-state impurity fraction from wall sputtering.

    Returns (species_symbol, f_z) where f_z = n_z/n_e.

    f_z = Y(3*T_edge) * (A_wall / V_plasma) * tau_ratio * f_screen
    """
    symbol = wall_material.value
    Y = compute_sputtering_yield(T_edge_keV, wall_material)
    f_z = Y * (fw_area / jnp.maximum(plasma_volume, 1.0)) * tau_ratio * f_screen
    return symbol, f_z


# ---------------------------------------------------------------------------
# Coronal equilibrium cooling curves L_z(T_e) [W·m³]
# ---------------------------------------------------------------------------
# Piecewise power-law fits to Post-Jensen / Mavrin data.
# Each species: list of (T_min_keV, T_max_keV, coefficient, exponent)
# such that L_z = coeff * T_e^exponent in that range.
# Units: L_z in W·m³, T_e in keV.
_COOLING_CURVES: dict[str, list[tuple[float, float, float, float]]] = {
    "W": [
        (0.01, 0.1, 5.0e-31, -1.0),  # strong radiation at low T
        (0.1, 1.0, 1.5e-31, 0.5),  # rising through ionization stages
        (1.0, 10.0, 5.0e-31, 0.0),  # plateau
        (10.0, 100.0, 5.0e-31, -0.5),  # slow decline
    ],
    "C": [
        (0.01, 0.01, 1.0e-31, 0.0),  # fully ionized above ~0.3 keV
        (0.01, 0.1, 3.0e-32, -0.5),
        (0.1, 0.5, 1.0e-32, 0.5),
        (0.5, 100.0, 1.0e-36, 0.0),  # fully stripped, negligible
    ],
    "Be": [
        (0.01, 0.05, 2.0e-32, -0.5),
        (0.05, 0.3, 5.0e-33, 0.5),
        (0.3, 100.0, 1.0e-36, 0.0),  # fully stripped
    ],
    "Mo": [
        (0.01, 0.1, 2.0e-31, -1.0),
        (0.1, 1.0, 8.0e-32, 0.5),
        (1.0, 10.0, 2.0e-31, 0.0),
        (10.0, 100.0, 2.0e-31, -0.5),
    ],
    "Si": [
        (0.01, 0.1, 1.0e-31, -0.5),
        (0.1, 1.0, 3.0e-32, 0.5),
        (1.0, 100.0, 5.0e-33, 0.0),  # mostly stripped
    ],
    "Li": [
        # Li (Z=3): fully ionized above ~0.1 keV, negligible line radiation
        (0.01, 0.05, 1.0e-33, -0.5),
        (0.05, 0.1, 5.0e-34, 0.0),
        (0.1, 100.0, 1.0e-38, 0.0),  # ~zero at reactor temperatures
    ],
    "Ne": [
        (0.01, 0.1, 5.0e-32, -0.5),
        (0.1, 0.5, 2.0e-32, 0.5),
        (0.5, 2.0, 4.0e-32, 0.0),  # Ne-like shell peak
        (2.0, 100.0, 4.0e-32, -1.0),  # decline
    ],
    "Ar": [
        (0.01, 0.1, 1.0e-31, -0.5),
        (0.1, 1.0, 5.0e-32, 0.5),
        (1.0, 5.0, 1.0e-31, 0.0),  # broad peak
        (5.0, 100.0, 1.0e-31, -0.5),  # decline
    ],
}


def cooling_rate(species: str, T_e_keV: float) -> float:
    """Coronal equilibrium radiative loss rate L_z [W·m³].

    Piecewise power-law fits for each species. Returns L_z at given T_e.
    For species not in the table, returns 0.
    """
    if species not in _COOLING_CURVES:
        return 0.0
    # Map SiC wall material to Si cooling curve
    lookup = "Si" if species == "SiC" else species
    if lookup not in _COOLING_CURVES:
        return 0.0

    segments = _COOLING_CURVES[lookup]
    T_e = jnp.maximum(T_e_keV, 0.01)

    # Evaluate all segments and select the matching one
    result = 0.0
    for T_min, T_max, coeff, exponent in segments:
        in_range = (T_e >= T_min) & (T_e < T_max)
        L_z = coeff * T_e**exponent
        result = jnp.where(in_range, L_z, result)

    return result


def compute_p_line(
    n_e: float,
    T_e: float,
    impurities: ImpurityMix | None,
    volume: float,
) -> float:
    """Impurity line radiation power [MW].

    P_line = n_e² × Σ(f_z × L_z(T_e)) × V
    """
    if impurities is None:
        return 0.0
    total_Lz = 0.0
    all_species = {**impurities.wall_derived, **impurities.seeded}
    for species, f_z in all_species.items():
        total_Lz = total_Lz + f_z * cooling_rate(species, T_e)
    # Scale n_e to avoid float32 overflow (n_e~1e20, n_e^2~1e40 > float32 max)
    n_e_20 = n_e * 1e-20
    return total_Lz * n_e_20**2 * volume * 1e34  # 1e40 * 1e-6 = 1e34


def compute_p_rad(
    n_e: float,
    T_e: float,
    Z_eff: float,
    volume: float,
    B: float = 0.0,
    impurities: ImpurityMix | None = None,
) -> float:
    """Plasma radiation power (bremsstrahlung + synchrotron + line).

    Calculated from plasma parameters by default, can be overridden.
    P_brem = 5.35e-37 * n_e^2 * Z_eff * sqrt(T_e) * V  [W], T_e in keV, n_e in m^-3
    P_sync = 6.2e-22 * n_e * T_e^2 * B^2 * V  [W] (MFE only)
    P_line = n_e^2 * Σ(f_z * L_z(T_e)) * V  [W] (impurity line radiation)
    """
    # Scale n_e to avoid float32 overflow (n_e~1e20, n_e^2~1e40 > float32 max)
    n_e_20 = n_e * 1e-20
    p_brem = 5.35e3 * n_e_20**2 * Z_eff * jnp.sqrt(T_e) * volume * 1e-6  # -> MW
    p_sync = 6.2e-2 * n_e_20 * T_e**2 * B**2 * volume * 1e-6  # -> MW
    p_line = compute_p_line(n_e, T_e, impurities, volume)
    return p_brem + p_sync + p_line


def mfe_forward_power_balance(
    p_fus: float,
    fuel: Fuel,
    p_input: float,
    mn: float,
    eta_th: float,
    eta_p: float,
    eta_pin: float,
    eta_de: float,
    f_sub: float,
    f_dec: float,
    p_coils: float,
    p_cool: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    # Radiation: calculated from plasma params, or override with p_rad_override
    n_e: float = 1.0e20,
    T_e: float = 15.0,
    Z_eff: float = 1.5,
    plasma_volume: float = 500.0,
    B: float = 5.0,
    p_rad_override: float | None = None,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
    # Impurity line radiation model
    wall_material: WallMaterial | None = None,
    seeded_impurities: dict[str, float] | None = None,
    T_edge: float = 0.05,
    f_screen: float = 0.01,
    tau_ratio: float = 3.0,
    fw_area: float = 0.0,
) -> PowerTable:
    """MFE forward power balance: fusion power -> net electric.

    Source: pyFECONs power_balance.py + fusion-tea mfe_power_balance.sysml
    Radiation model: bremsstrahlung + synchrotron + impurity line radiation.
    """
    # Step 1: Ash/neutron split
    p_ash, p_neutron = ash_neutron_split(
        p_fus,
        fuel,
        dd_f_T,
        dd_f_He3,
        dhe3_dd_frac,
        dhe3_f_T,
        pb11_f_alpha_n,
        pb11_f_p_n,
    )

    # Step 2: Build impurity mix (if wall_material provided)
    impurities = None
    if wall_material is not None and p_rad_override is None:
        wall_derived = {}
        symbol, f_z = compute_impurity_fraction(
            wall_material,
            T_edge,
            fw_area,
            plasma_volume,
            f_screen,
            tau_ratio,
        )
        wall_derived[symbol] = f_z
        impurities = ImpurityMix(
            wall_derived=wall_derived,
            seeded=seeded_impurities or {},
        )
    elif seeded_impurities and p_rad_override is None:
        impurities = ImpurityMix(wall_derived={}, seeded=seeded_impurities)

    # Step 3: Radiation power (p_rad + p_transport = p_ash)
    if p_rad_override is not None:
        p_rad = p_rad_override
    else:
        p_rad = compute_p_rad(n_e, T_e, Z_eff, plasma_volume, B, impurities)
    p_rad = jnp.minimum(p_rad, p_ash)  # Can't radiate more than ash power
    p_transport = p_ash - p_rad

    # Step 4: Auxiliary power
    p_aux = p_trit + p_house

    # Step 5: DEC routing (operates on transport channel only)
    p_dee = f_dec * eta_de * p_transport
    p_dec_waste = f_dec * (1.0 - eta_de) * p_transport
    p_wall = (1.0 - f_dec) * p_transport

    # Step 6: Thermal power (neutrons + radiation + wall transport + heating + pumping)
    p_th = mn * p_neutron + p_rad + p_wall + p_input + eta_p * p_pump

    # Step 7: Thermal electric
    p_the = eta_th * p_th

    # Step 8: Gross electric
    p_et = p_dee + p_the

    # Step 9: Lost power
    p_loss = (p_th - p_the) + p_dec_waste

    # Step 10: Subsystem power
    p_sub = f_sub * p_et

    # Step 11: Scientific Q
    q_sci = p_fus / p_input

    # Step 12: Engineering Q
    recirculating = (
        p_coils + p_pump + p_sub + p_aux + p_cool + p_cryo + p_input / eta_pin
    )
    q_eng = p_et / recirculating

    # Step 12: Net electric
    rec_frac = 1.0 / q_eng
    p_net = (1.0 - rec_frac) * p_et

    return PowerTable(
        p_fus=p_fus,
        p_ash=p_ash,
        p_neutron=p_neutron,
        p_rad=p_rad,
        p_wall=p_wall,
        p_dee=p_dee,
        p_dec_waste=p_dec_waste,
        p_th=p_th,
        p_the=p_the,
        p_et=p_et,
        p_loss=p_loss,
        p_net=p_net,
        p_pump=p_pump,
        p_sub=p_sub,
        p_aux=p_aux,
        p_coils=p_coils,
        p_cool=p_cool,
        p_cryo=p_cryo,
        p_target=0.0,
        q_sci=q_sci,
        q_eng=q_eng,
        rec_frac=rec_frac,
    )


def mfe_inverse_power_balance(
    p_net_target: float,
    fuel: Fuel,
    p_input: float,
    mn: float,
    eta_th: float,
    eta_p: float,
    eta_pin: float,
    eta_de: float,
    f_sub: float,
    f_dec: float,
    p_coils: float,
    p_cool: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    # Radiation: calculated from plasma params, or override
    n_e: float = 1.0e20,
    T_e: float = 15.0,
    Z_eff: float = 1.5,
    plasma_volume: float = 500.0,
    B: float = 5.0,
    p_rad_override: float | None = None,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
    # Impurity line radiation model
    wall_material: WallMaterial | None = None,
    seeded_impurities: dict[str, float] | None = None,
    T_edge: float = 0.05,
    f_screen: float = 0.01,
    tau_ratio: float = 3.0,
    fw_area: float = 0.0,
) -> float:
    """Inverse MFE power balance: target net electric -> required fusion power.

    Closed-form linear inversion (no iteration needed).
    p_rad is constant w.r.t. p_fus (depends on plasma params, not fusion power),
    so it enters as a constant term in the linear inversion.
    Source: fusion-tea Inverse MFE Power Balance Calc
    """
    # Step 1: Build impurity mix (if wall_material provided)
    impurities = None
    if wall_material is not None and p_rad_override is None:
        wall_derived = {}
        symbol, f_z = compute_impurity_fraction(
            wall_material,
            T_edge,
            fw_area,
            plasma_volume,
            f_screen,
            tau_ratio,
        )
        wall_derived[symbol] = f_z
        impurities = ImpurityMix(
            wall_derived=wall_derived,
            seeded=seeded_impurities or {},
        )
    elif seeded_impurities and p_rad_override is None:
        impurities = ImpurityMix(wall_derived={}, seeded=seeded_impurities)

    # Step 2: Radiation power (constant w.r.t. p_fus)
    if p_rad_override is not None:
        p_rad = p_rad_override
    else:
        p_rad = compute_p_rad(n_e, T_e, Z_eff, plasma_volume, B, impurities)

    # Step 2: Ash fraction from fuel type (use p_fus=1.0 to get the fraction)
    p_ash_unit, _ = ash_neutron_split(
        1.0, fuel, dd_f_T, dd_f_He3, dhe3_dd_frac, dhe3_f_T, pb11_f_alpha_n, pb11_f_p_n
    )
    ash_frac = p_ash_unit
    neutron_frac = 1.0 - ash_frac

    # Step 3: Linearize forward chain coefficients
    # Thermal power per unit p_fus
    c_th = mn * neutron_frac + (1.0 - f_dec) * ash_frac

    # Constant thermal power (radiation + heating + pumping)
    # p_rad - (1-f_dec)*p_rad simplifies to f_dec*p_rad
    c_th0 = f_dec * p_rad + p_input + eta_p * p_pump

    # DEC electric per unit p_fus
    c_dee = f_dec * eta_de * ash_frac
    c_dee0 = -f_dec * eta_de * p_rad

    # Gross electric: p_et = c_et * p_fus + c_et0
    c_et = c_dee + eta_th * c_th
    c_et0 = c_dee0 + eta_th * c_th0

    # Recirculating (p_fus-dependent): p_sub = f_sub * p_et
    c_den = f_sub * c_et

    # Recirculating (constant loads)
    p_aux = p_trit + p_house
    c_den0 = (
        p_coils + p_pump + f_sub * c_et0 + p_aux + p_cool + p_cryo + p_input / eta_pin
    )

    # Step 4: Solve p_net = (c_et - c_den) * p_fus + (c_et0 - c_den0)
    p_fus = (p_net_target - c_et0 + c_den0) / (c_et - c_den)
    return p_fus


# ---------------------------------------------------------------------------
# IFE Power Balance
# ---------------------------------------------------------------------------


def ife_forward_power_balance(
    p_fus: float,
    fuel: Fuel,
    p_implosion: float,
    p_ignition: float,
    mn: float,
    eta_th: float,
    eta_p: float,
    eta_pin1: float,
    eta_pin2: float,
    f_sub: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    p_target: float,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
) -> PowerTable:
    """IFE forward power balance: fusion power -> net electric.

    Key differences from MFE:
    - No DEC (all ash thermalizes)
    - Split driver: implosion + ignition, each with own wall-plug efficiency
    - Target factory power in recirculating
    - No coil power or coil cooling
    - No plasma confinement radiation (p_rad=0)
    Source: pyFECONs power_balance.py (IFE branch)
    """
    p_input = p_implosion + p_ignition

    # Step 1: Ash/neutron split
    p_ash, p_neutron = ash_neutron_split(
        p_fus,
        fuel,
        dd_f_T,
        dd_f_He3,
        dhe3_dd_frac,
        dhe3_f_T,
        pb11_f_alpha_n,
        pb11_f_p_n,
    )

    # Step 2: No DEC in IFE — all ash thermalizes
    p_rad = 0.0
    p_wall = p_ash
    p_dee = 0.0
    p_dec_waste = 0.0

    # Step 3: Auxiliary power
    p_aux = p_trit + p_house

    # Step 4: Thermal power (all ash thermalizes + neutrons + heating + pumping)
    p_th = mn * p_neutron + p_ash + p_input + eta_p * p_pump

    # Step 5: Thermal electric
    p_the = eta_th * p_th

    # Step 6: Gross electric (no DEC)
    p_et = p_the

    # Step 7: Lost power
    p_loss = p_th - p_the

    # Step 8: Subsystem power
    p_sub = f_sub * p_et

    # Step 9: Scientific Q
    q_sci = p_fus / p_input

    # Step 10: Engineering Q — split driver efficiencies
    recirculating = (
        p_target
        + p_pump
        + p_sub
        + p_aux
        + p_cryo
        + p_implosion / eta_pin1
        + p_ignition / eta_pin2
    )
    q_eng = p_et / recirculating

    # Step 11: Net electric
    rec_frac = 1.0 / q_eng
    p_net = (1.0 - rec_frac) * p_et

    return PowerTable(
        p_fus=p_fus,
        p_ash=p_ash,
        p_neutron=p_neutron,
        p_rad=p_rad,
        p_wall=p_wall,
        p_dee=p_dee,
        p_dec_waste=p_dec_waste,
        p_th=p_th,
        p_the=p_the,
        p_et=p_et,
        p_loss=p_loss,
        p_net=p_net,
        p_pump=p_pump,
        p_sub=p_sub,
        p_aux=p_aux,
        p_coils=0.0,
        p_cool=0.0,
        p_cryo=p_cryo,
        p_target=p_target,
        q_sci=q_sci,
        q_eng=q_eng,
        rec_frac=rec_frac,
    )


def ife_inverse_power_balance(
    p_net_target: float,
    fuel: Fuel,
    p_implosion: float,
    p_ignition: float,
    mn: float,
    eta_th: float,
    eta_p: float,
    eta_pin1: float,
    eta_pin2: float,
    f_sub: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    p_target: float,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
) -> float:
    """Inverse IFE power balance: target net electric -> required fusion power.

    Closed-form linear inversion (same approach as MFE inverse).
    """
    p_input = p_implosion + p_ignition

    # Ash fraction
    ash_frac, _ = ash_neutron_split(
        1.0,
        fuel,
        dd_f_T,
        dd_f_He3,
        dhe3_dd_frac,
        dhe3_f_T,
        pb11_f_alpha_n,
        pb11_f_p_n,
    )
    neutron_frac = 1.0 - ash_frac

    # All ash thermalizes (no DEC in IFE)
    c_th = mn * neutron_frac + ash_frac
    c_th0 = p_input + eta_p * p_pump

    # Gross electric = thermal electric (no DEC)
    c_et = eta_th * c_th
    c_et0 = eta_th * c_th0

    # Recirculating (p_fus-dependent part): p_sub = f_sub * p_et
    c_den = f_sub * c_et

    # Recirculating (constant loads)
    p_aux = p_trit + p_house
    c_den0 = (
        p_target
        + p_pump
        + f_sub * c_et0
        + p_aux
        + p_cryo
        + p_implosion / eta_pin1
        + p_ignition / eta_pin2
    )

    p_fus = (p_net_target - c_et0 + c_den0) / (c_et - c_den)
    return p_fus


# ---------------------------------------------------------------------------
# MIF Power Balance
# ---------------------------------------------------------------------------


def mif_forward_power_balance(
    p_fus: float,
    fuel: Fuel,
    p_driver: float,
    mn: float,
    eta_th: float,
    eta_p: float,
    eta_pin: float,
    f_sub: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    p_target: float,
    p_coils: float = 0.0,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
) -> PowerTable:
    """MIF forward power balance: fusion power -> net electric.

    MIF uses pulsed magnetic compression. Key differences:
    - Single driver (pulsed power) with one wall-plug efficiency
    - Target/liner factory power in recirculating
    - Optional small guide-field coils
    - No DEC (all ash thermalizes)
    - No plasma confinement radiation
    """
    # Step 1: Ash/neutron split
    p_ash, p_neutron = ash_neutron_split(
        p_fus,
        fuel,
        dd_f_T,
        dd_f_He3,
        dhe3_dd_frac,
        dhe3_f_T,
        pb11_f_alpha_n,
        pb11_f_p_n,
    )

    # Step 2: No DEC — all ash thermalizes
    p_rad = 0.0
    p_wall = p_ash
    p_dee = 0.0
    p_dec_waste = 0.0

    # Step 3: Auxiliary power
    p_aux = p_trit + p_house

    # Step 4: Thermal power
    p_th = mn * p_neutron + p_ash + p_driver + eta_p * p_pump

    # Step 5: Thermal electric
    p_the = eta_th * p_th

    # Step 6: Gross electric (no DEC)
    p_et = p_the

    # Step 7: Lost power
    p_loss = p_th - p_the

    # Step 8: Subsystem power
    p_sub = f_sub * p_et

    # Step 9: Scientific Q
    q_sci = p_fus / p_driver

    # Step 10: Engineering Q
    recirculating = (
        p_target + p_coils + p_pump + p_sub + p_aux + p_cryo + p_driver / eta_pin
    )
    q_eng = p_et / recirculating

    # Step 11: Net electric
    rec_frac = 1.0 / q_eng
    p_net = (1.0 - rec_frac) * p_et

    return PowerTable(
        p_fus=p_fus,
        p_ash=p_ash,
        p_neutron=p_neutron,
        p_rad=p_rad,
        p_wall=p_wall,
        p_dee=p_dee,
        p_dec_waste=p_dec_waste,
        p_th=p_th,
        p_the=p_the,
        p_et=p_et,
        p_loss=p_loss,
        p_net=p_net,
        p_pump=p_pump,
        p_sub=p_sub,
        p_aux=p_aux,
        p_coils=p_coils,
        p_cool=0.0,
        p_cryo=p_cryo,
        p_target=p_target,
        q_sci=q_sci,
        q_eng=q_eng,
        rec_frac=rec_frac,
    )


def mif_inverse_power_balance(
    p_net_target: float,
    fuel: Fuel,
    p_driver: float,
    mn: float,
    eta_th: float,
    eta_p: float,
    eta_pin: float,
    f_sub: float,
    p_pump: float,
    p_trit: float,
    p_house: float,
    p_cryo: float,
    p_target: float,
    p_coils: float = 0.0,
    dd_f_T: float = DD_F_T_DEFAULT,
    dd_f_He3: float = DD_F_HE3_DEFAULT,
    dhe3_dd_frac: float = 0.07,
    dhe3_f_T: float = 0.97,
    pb11_f_alpha_n: float = 0.0,
    pb11_f_p_n: float = 0.0,
) -> float:
    """Inverse MIF power balance: target net electric -> required fusion power.

    Closed-form linear inversion.
    """
    # Ash fraction
    ash_frac, _ = ash_neutron_split(
        1.0,
        fuel,
        dd_f_T,
        dd_f_He3,
        dhe3_dd_frac,
        dhe3_f_T,
        pb11_f_alpha_n,
        pb11_f_p_n,
    )
    neutron_frac = 1.0 - ash_frac

    # All ash thermalizes
    c_th = mn * neutron_frac + ash_frac
    c_th0 = p_driver + eta_p * p_pump

    # Gross electric = thermal electric
    c_et = eta_th * c_th
    c_et0 = eta_th * c_th0

    # Recirculating
    c_den = f_sub * c_et

    p_aux = p_trit + p_house
    c_den0 = (
        p_target
        + p_coils
        + p_pump
        + f_sub * c_et0
        + p_aux
        + p_cryo
        + p_driver / eta_pin
    )

    p_fus = (p_net_target - c_et0 + c_den0) / (c_et - c_den)
    return p_fus
