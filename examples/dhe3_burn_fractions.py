"""Compute burn fractions for D-D-bred T and He-3 in a D-He-3 plasma.

In a thermal D-He-3 plasma, D-D side reactions produce tritium and He-3
that can re-burn in D-T and D-He-3 fusion reactions. The fraction of
produced tritons (or He-3 atoms) that fuse before being exhausted is

    f_X = (n_D <sigma v>_X * tau_p) / (1 + n_D <sigma v>_X * tau_p)

where tau_p is the particle confinement time of species X. This script
computes f_T and f_He3 as functions of T and tau_p, using Bosch-Hale
cross-sections, to anchor the 1costingfe defaults dhe3_f_T and dhe3_f_He3.

Convention: "tau_p" here is the species-particle confinement time. In
practice tau_p ~ a few * tau_E for the bulk fuel; ash species (T, alpha)
may have shorter or longer confinement depending on geometry. We sweep
tau_p over a range and report which value reproduces the literature
reference points.

References:
- Bosch-Hale, Nucl. Fusion 32, 611 (1992)
- Galambos & Peng, Fusion Tech. 8, 1599 (1985) - D-He3 reactor designs
- Stott, PPCF 47 BR3 (2005) - D-He3 feasibility review
"""

import numpy as np


# ============================================================
# Bosch-Hale (1992) thermal reactivities. T in keV, <sigma v> in m^3/s.
# ============================================================
def _bh(T, BG, mrc2, C1, C2, C3, C4, C5, C6, C7):
    num = T * (C2 + T * (C4 + T * C6))
    den = 1 + T * (C3 + T * (C5 + T * C7))
    theta = T / (1 - num / den)
    xi = (BG**2 / (4 * theta)) ** (1.0 / 3)
    return C1 * theta * np.sqrt(xi / (mrc2 * T**3)) * np.exp(-3 * xi) * 1e-6


def sigv_dt(T):
    """D + T -> n + 4He. Bosch-Hale 1992 parameters."""
    return _bh(
        T,
        34.3827,
        1124656,
        1.17302e-9,
        1.51361e-2,
        7.51886e-2,
        4.60643e-3,
        1.35000e-2,
        -1.06750e-4,
        1.36600e-5,
    )


def sigv_dhe3(T):
    """D + 3He -> p + 4He. Bosch-Hale 1992 parameters."""
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


def burn_fraction(n_D, sigv_X, tau_p):
    """Burn fraction for species X in plasma with deuterium density n_D."""
    R = n_D * sigv_X * tau_p
    return R / (1.0 + R)


# ============================================================
# Reference plasma: D-He-3 50/50 mix, n_e = 1e20 m^-3
# ============================================================
# Quasineutrality: n_e = n_D + 2 n_He3 (He-3 has charge +2).
# 50/50 mix means n_D = n_He3.
# So n_e = 3 n_D, n_D = n_e / 3.
N_E = 1.0e20  # m^-3
N_D = N_E / 3.0  # 3.33e19 m^-3 for 50/50

if __name__ == "__main__":
    print("=" * 72)
    print("Burn fractions for D-D-bred T and He-3 in a D-He-3 plasma")
    print("=" * 72)
    print(f"Reference: 50/50 D/He-3, n_e = {N_E:.1e} m^-3, n_D = {N_D:.2e} m^-3")
    print()

    print("Cross-sections at the operating point:")
    print(f"  {'T (keV)':>8} {'<sv>_DT (m^3/s)':>20} {'<sv>_DHe3 (m^3/s)':>22}")
    for T in [50, 70, 100, 150]:
        print(f"  {T:>8} {sigv_dt(T):>20.3e} {sigv_dhe3(T):>22.3e}")

    print()
    print("Burn fractions vs particle confinement time tau_p (T_i = 70 keV):")
    print(
        f"  {'tau_p (s)':>10} {'n_D <sv>_DT tau':>18} {'f_T':>8}"
        f" {'n_D <sv>_DHe3 tau':>20} {'f_He3':>8}"
    )
    print("  " + "-" * 70)
    T = 70.0
    sv_dt = sigv_dt(T)
    sv_dhe3 = sigv_dhe3(T)
    for tau_p in [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
        rt = N_D * sv_dt * tau_p
        rhe3 = N_D * sv_dhe3 * tau_p
        f_T = rt / (1 + rt)
        f_He3 = rhe3 / (1 + rhe3)
        print(f"  {tau_p:>10.1f} {rt:>18.3f} {f_T:>8.3f} {rhe3:>20.3f} {f_He3:>8.3f}")

    print()
    print("Burn fractions vs T (tau_p = 10 s):")
    print(f"  {'T (keV)':>8} {'f_T':>8} {'f_He3':>8}")
    for T in [50, 70, 100, 150]:
        f_T = burn_fraction(N_D, sigv_dt(T), 10.0)
        f_He3 = burn_fraction(N_D, sigv_dhe3(T), 10.0)
        print(f"  {T:>8} {f_T:>8.3f} {f_He3:>8.3f}")

    print()
    print("=" * 72)
    print("Interpretation")
    print("=" * 72)
    print("""
At T_i = 70 keV with n_D = 3.3e19 m^-3 (n_e = 1e20, 50/50 mix):
  <sv>_DT  = ~8e-22 m^3/s   (much larger than D-D cross-section)
  <sv>_DHe3 = ~6e-23 m^3/s  (~13x smaller than D-T at this T)

Burn fraction f_X = n_D * <sv>_X * tau_p / (1 + n_D * <sv>_X * tau_p)
depends sensitively on the particle confinement time tau_p:

  tau_p =   1 s:  f_T = 0.03,  f_He3 = 0.002  (single-pass burnup)
  tau_p =  10 s:  f_T = 0.21,  f_He3 = 0.020  (modest pile-up)
  tau_p = 100 s:  f_T = 0.73,  f_He3 = 0.17   (heavy pile-up)
  tau_p =1000 s:  f_T = 0.96,  f_He3 = 0.67   (asymptotic)

The 1costingfe defaults dhe3_f_T = dhe3_f_He3 = 0.97 are only
appropriate in the asymptotic limit (tau_p -> very large), which
corresponds to recovering and re-injecting the unburned ash.
For typical single-pass reactor confinement, both fractions are
much smaller, and f_He3 is always lower than f_T because <sv>_DHe3
is smaller than <sv>_DT at relevant temperatures.

Recommendations for 1costingfe defaults at T_i = 70-100 keV with
typical reactor confinement (tau_p ~ 5-10 tau_E ~ 5-30 s) and ash
recycling:
  dhe3_f_T   ~ 0.5  (literature reference: Galambos & Peng 1985,
                     Stott 2005, Cano et al. 1976)
  dhe3_f_He3 ~ 0.1  (consistent with single-pass plus partial recycle)

The current default 0.97 should be treated as an upper-bound
sensitivity case, not a base case.
""")
