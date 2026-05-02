"""D-He3 fuel mix optimization for a Helion-like (magneto-inertial) plasma.

In a magneto-inertial plasmoid the plasma is held too briefly for D-D-bred
tritium to fuse, so the secondary D-T channel is closed. This script computes
the per-shot energy split between extractable charged power, bremsstrahlung,
and neutrons as a function of the D/He-3 density ratio and temperature, and
finds the operating point that minimizes bremsstrahlung losses.

Assumptions:
- f_T = 0: bred tritium exhausted before D-T fusion completes (MIF timescale)
- f_He3 = 0: single-pass plasma physics; inter-shot He-3 recovery is a
  system-level effect handled separately in the cost model, not credited here
- Bosch-Hale cross sections (Nucl. Fusion 32, 611, 1992)
- Bremsstrahlung from NRL Plasma Formulary with relativistic + e-e corrections
  (Rider 1995)
- Thermal equilibrium (T_e = T_i)
- No alpha-ash buildup in Z_eff
- Synchrotron and impurity-line radiation not included

Used by the "Direct Energy Conversion and the Cost Floor" post to set the
Helion-likely operating point and to derive the corresponding 1costingfe
inputs (dhe3_dd_frac and f_rad_fus_dhe3).
"""

import numpy as np
from scipy.optimize import minimize_scalar


# ============================================================
# Bosch-Hale (1992) thermal reactivities. T in keV, <sigma v> in m^3/s.
# ============================================================
def _bh(T, BG, mrc2, C1, C2, C3, C4, C5, C6, C7):
    num = T * (C2 + T * (C4 + T * C6))
    den = 1 + T * (C3 + T * (C5 + T * C7))
    theta = T / (1 - num / den)
    xi = (BG**2 / (4 * theta)) ** (1.0 / 3)
    return C1 * theta * np.sqrt(xi / (mrc2 * T**3)) * np.exp(-3 * xi) * 1e-6


def sigv_dhe3(T):
    """D + 3He -> p + 4He."""
    return _bh(
        T,
        68.7508,
        1124572,
        5.51036e-10,
        6.41918e-3,
        -2.02896e-3,
        -1.91080e-5,
        1.35776e-4,
        0,
        0,
    )


def sigv_dd_n(T):
    """D + D -> n + 3He branch."""
    return _bh(
        T, 31.3970, 937814, 5.43360e-12, 5.85778e-3, 7.68222e-3, 0, -2.96400e-6, 0, 0
    )


def sigv_dd_p(T):
    """D + D -> p + T branch."""
    return _bh(
        T, 31.3970, 937814, 5.65718e-12, 3.41267e-3, 1.99167e-3, 0, 1.05060e-5, 0, 0
    )


def sigv_dd_total(T):
    return sigv_dd_n(T) + sigv_dd_p(T)


# ============================================================
# Per-event energies (MeV). Helion-like: T is exhausted before it fuses,
# so neither D-T nor secondary D-He3 burnup contributes at the per-shot level.
# Each D-D event contributes only its primary kinetic energy:
#   D(d,p)T:    p (3.02 MeV) + T (1.01 MeV) = 4.03 MeV charged
#   D(d,n)3He:  n (2.45 MeV) + 3He (0.82 MeV) = 3.27 MeV total, 2.45 MeV neutron
# ============================================================
E_DHE3 = 18.35  # all charged (4He + p)
E_N_DD = 0.5 * 2.45  # neutron from primary D(d,n)3He branch only
E_C_DD = 0.5 * 0.82 + 0.5 * 4.03  # primary charged (3He + p + T)
E_DD_TOT = E_N_DD + E_C_DD


# ============================================================
# Bremsstrahlung — non-relativistic NRL plus relativistic + e-e corrections
# ============================================================
def brem_factor_rel(T_keV, Zeff):
    """Relativistic e-i + e-e correction to non-relativistic NRL brem.

    From Rider (1995) / Wesson Tokamaks. m_e c^2 = 511 keV.
    """
    tau = T_keV / 511.0
    rel_ei = 1 + 0.7936 * tau + 1.874 * tau**2
    ee_correction = 1.5 * tau / Zeff
    return rel_ei + ee_correction


# ============================================================
# Energy split at given mix r = n_3He / n_D and temperature
# ============================================================
def fractions(r, T_keV, n_e=1e20):
    """Compute neutron / bremsstrahlung / extractable fractions of fusion power.

    Inputs:
        r = n_3He / n_D (so D-rich means r < 1)
        T_keV = ion/electron temperature (assumed equal)
        n_e = electron density (m^-3); cancels out for fractions

    Returns dict with f_n, f_brem, f_ext, f_DD (reaction fraction), Zeff.
    """
    Zeff = (1 + 4 * r) / (1 + 2 * r)  # quasineutrality, He-3 is +2
    n_D = n_e / (1 + 2 * r)
    n_He3 = r * n_e / (1 + 2 * r)

    R_DHe3 = n_D * n_He3 * sigv_dhe3(T_keV)
    R_DD = 0.5 * n_D**2 * sigv_dd_total(T_keV)

    MeV_to_J = 1.602e-13
    P_fus = (R_DHe3 * E_DHE3 + R_DD * E_DD_TOT) * MeV_to_J  # W/m^3
    P_n = R_DD * E_N_DD * MeV_to_J
    P_c = (R_DHe3 * E_DHE3 + R_DD * E_C_DD) * MeV_to_J

    # NRL bremsstrahlung volume emission coefficient (W/m^3, T in eV)
    P_brem_NR = 1.69e-38 * Zeff * n_e**2 * np.sqrt(T_keV * 1000)
    P_brem = P_brem_NR * brem_factor_rel(T_keV, Zeff)

    f_n = P_n / P_fus
    f_brem = P_brem / P_fus
    f_ext = max(P_c / P_fus - f_brem, 0.0)
    f_DD = R_DD / (R_DHe3 + R_DD)

    return dict(f_n=f_n, f_brem=f_brem, f_ext=f_ext, f_DD=f_DD, Zeff=Zeff)


def find_minimum_brem(T_keV, r_bounds=(0.05, 5.0)):
    """Find the mix r = n_3He/n_D that minimizes bremsstrahlung at fixed T."""
    res = minimize_scalar(
        lambda r: fractions(r, T_keV)["f_brem"],
        bounds=r_bounds,
        method="bounded",
    )
    r = float(res.x)
    return r, fractions(r, T_keV)


def find_minimum_loss(T_keV, r_bounds=(0.05, 5.0)):
    """Find the mix that minimizes (neutrons + bremsstrahlung) at fixed T."""
    res = minimize_scalar(
        lambda r: fractions(r, T_keV)["f_n"] + fractions(r, T_keV)["f_brem"],
        bounds=r_bounds,
        method="bounded",
    )
    r = float(res.x)
    return r, fractions(r, T_keV)


# ============================================================
# Reporting
# ============================================================
if __name__ == "__main__":
    print("=" * 75)
    print("Helion-like D-He3 plasma: per-shot energy split")
    print("(magneto-inertial timescale: T exhausted, no secondary burnup)")
    print("=" * 75)

    print("\nMix sweep at T = 100 keV (n_e = 1e20 m^-3):")
    print(
        f"  {'r':>5} {'n_D/n_3He':>10} {'f_DD':>7}"
        f" {'f_n':>6} {'f_brem':>8} {'f_ext':>8}"
    )
    print("  " + "-" * 55)
    for r in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        f = fractions(r, 100)
        print(
            f"  {r:>5.2f} {1 / r:>10.2f}"
            f" {f['f_DD'] * 100:>6.1f}%"
            f" {f['f_n'] * 100:>5.1f}%"
            f" {f['f_brem'] * 100:>7.1f}%"
            f" {f['f_ext'] * 100:>7.1f}%"
        )

    print("\nTemperature sweep at r = 0.31 (D-rich, brem-minimum at 100 keV):")
    print(f"  {'T (keV)':>8} {'f_DD':>7} {'f_n':>6} {'f_brem':>8} {'f_ext':>8}")
    print("  " + "-" * 50)
    for T in [50, 70, 100, 150, 200]:
        f = fractions(0.31, T)
        print(
            f"  {T:>8} {f['f_DD'] * 100:>6.1f}%"
            f" {f['f_n'] * 100:>5.1f}%"
            f" {f['f_brem'] * 100:>7.1f}%"
            f" {f['f_ext'] * 100:>7.1f}%"
        )

    print()
    print("=" * 75)
    print("Brem-minimum mix at fixed temperature")
    print("=" * 75)
    print(
        f"  {'T (keV)':>8} {'r_opt':>7} {'n_D/n_3He':>10}"
        f" {'f_n':>6} {'f_brem':>8} {'f_ext':>8}"
    )
    print("  " + "-" * 60)
    for T in [50, 70, 100, 150, 200]:
        r, f = find_minimum_brem(T)
        print(
            f"  {T:>8} {r:>7.2f} {1 / r:>10.2f}"
            f" {f['f_n'] * 100:>5.1f}%"
            f" {f['f_brem'] * 100:>7.1f}%"
            f" {f['f_ext'] * 100:>7.1f}%"
        )

    print()
    print("=" * 75)
    print("Helion-likely operating point: brem-minimum at T = 100 keV")
    print("=" * 75)
    r_opt, f = find_minimum_brem(100)
    print(f"  r = n_3He / n_D     = {r_opt:.3f}")
    print(f"  n_D / n_3He         = {1 / r_opt:.2f}")
    print("  Energy split:")
    print(f"    Extractable       = {f['f_ext'] * 100:.1f}%")
    print(f"    Bremsstrahlung    = {f['f_brem'] * 100:.1f}%")
    print(f"    Neutrons          = {f['f_n'] * 100:.1f}%")
    print(f"  D-D reaction frac   = {f['f_DD'] * 100:.1f}%")
    print(f"  Zeff                = {f['Zeff']:.2f}")
    print()
    print("  1costingfe inputs:")
    print(f"    dhe3_dd_frac      = {f['f_DD']:.3f}")
    print(f"    f_rad_fus_dhe3    = {f['f_brem']:.3f}")
