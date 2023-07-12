import math

import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols


def find_TF_thickness(n_TF, Ri, Re, R0, B0, A, d):
    flag_plot_tf = 1
    S_amm = 667e6  # Steel allowable limit
    R_VV = Ri * 1.05  # Vacuum vessel radius
    S_VV = 90e6  # Vacuum vessel steel limit
    Mu_0 = 4 * math.pi * 10**-7
    theta_TF = 2 * math.pi / n_TF
    dx_case = 2 * Ri * math.tan(theta_TF / 2)  # Case width
    dr_plasma_side = R0 * 2 / 3 * 1e-2
    gap = R0 * 2 / 3 * 1e-2
    NI = (2 * math.pi * R0 * B0 / Mu_0) / n_TF  # Total TF current [MA]
    sf = 1.08
    BI_TF = Mu_0 * n_TF * NI / (2 * math.pi * Ri) * sf  # Toroidal field peak
    Iop = 70e3
    n_spire = math.floor(NI / Iop)
    L = (
        Mu_0
        * R0
        * (n_TF * n_spire) ** 2
        * (1 - math.sqrt(1 - (R0 - Ri) / R0))
        / n_TF
        * 1.1
    )
    E = 1 / 2 * L * n_TF * Iop**2 * 1e-6
    V_MAX = (7 * R0 - 3) / 6 * 1.1e3  # [V]
    Tau_discharge1 = L * Iop / V_MAX  # [s] - scarico in gruppi di tre bobine
    Tau_discharge2 = B0 * NI * n_TF * (R0 / A) ** 2 / (R_VV * S_VV)
    Tau_discharge = max([Tau_discharge1, Tau_discharge2, 4])
    # materials
    E_jckt = 205
    E_cbl_HTS = 120
    E_tins = 12
    E_cbl_LTS = 0.1
    E_case = 205
    # superconductor
    Ic_LTS, Jc_LTS = Ic_Nb3Sn_WST(BI_TF, 1)
    Jc_HTS = Jc_REBCO(BI_TF) * 1e6
    if BI_TF < 15.0:
        S_SC_WP = NI / (Jc_LTS * 1e6)
        Cuin = S_SC_WP / 2  # non segregated copper fraction
        NB3SN = S_SC_WP / 2  # S/C fraction
        E_cbl = E_cbl_LTS
    else:
        S_SC_WP = NI / Jc_HTS
        Cuin = 0
        NB3SN = 0
        E_cbl = E_cbl_HTS
    # stabilizer
    Cus = NI / 200e6
    THS = Ths_TF_Nb3Sn(NI, BI_TF, Cus, Cuin, NB3SN, Tau_discharge)
    while THS > 250:
        Cus *= 1.05
        THS = Ths_TF_Nb3Sn(NI, BI_TF, Cus, Cuin, NB3SN, Tau_discharge)
    while THS < 250:
        Cus *= 0.99
        THS = Ths_TF_Nb3Sn(NI, BI_TF, Cus, Cuin, NB3SN, Tau_discharge)
    S_Cu_WP = Cus
    VF = 0.7
    S_cable = (S_Cu_WP + S_SC_WP) / VF
    cable_w = math.sqrt(4 * (S_cable / n_spire) / math.pi)
    # emag load
    ReRi = Re / Ri
    F0 = Mu_0 * n_TF * NI**2 / (4 * math.pi)
    FZ = F0 * math.log(ReRi) / 2
    Pm = BI_TF**2 / (2 * Mu_0)
    # steel WP
    CICC_w = symbols("CICC_w")
    tins = 0.001
    JT = CICC_w - cable_w
    Ke_cavo_rad = (
        2 * E_jckt * JT / CICC_w
        + 2 * tins * E_tins / CICC_w
        + (
            1 / (E_cbl * cable_w / cable_w)
            + 2 / (E_jckt * CICC_w / JT)
            + 2 / (E_tins * CICC_w / tins)
        )
        ** -1
    )
    K_jckt = 2 * JT / CICC_w * E_jckt
    dcr_jckt = K_jckt / Ke_cavo_rad
    saf = (CICC_w) / (CICC_w - cable_w)
    eqn = Eq(
        S_amm,
        Pm * saf * dcr_jckt
        + (FZ / 2) / (CICC_w**2 * n_spire - math.pi * cable_w**2 / 4),
    )
    CICC_w = solve(eqn, CICC_w)
    CICC_w = [c.evalf() for c in CICC_w if c > 0]
    CICC_w = round(max(CICC_w), 4)
    SF = 1.0
    JT = CICC_w - cable_w
    Cond_w = CICC_w + 2 * tins
    S_tot_WP = round((Cond_w**2 * n_spire) * SF, 6)
    S_steel_WP = S_tot_WP - (S_Cu_WP + S_SC_WP)
    # rectangular WP
    dx_WP = dx_case * 0.725
    dx_WP_min1 = dx_case * 0.525
    dx_WP_min2 = dx_case * 0.325
    h = S_tot_WP / dx_WP

    if 2 * (Ri - dr_plasma_side - h) * math.tan(theta_TF / 2) >= (dx_WP + 2 * gap):
        Rj = Ri - dr_plasma_side - h
        Rj_ = Rj
        Rj__ = Rj_
    else:
        Rj_ = (dx_WP + 2 * gap) / (2 * math.tan(theta_TF / 2))
        h_ = Ri + dr_plasma_side - Rj_
        A_ = S_tot_WP - (dx_WP * h_)
        h__ = A_ / dx_WP_min1
        Rj = Rj_ - h__
        Rj__ = Rj_
        if 2 * (Rj) * math.tan(theta_TF / 2) <= (dx_WP_min1 + 2 * gap):
            Rj_ = (dx_WP + 2 * gap) / (2 * math.tan(theta_TF / 2))
            Rj__ = (dx_WP_min1 + 2 * gap) / (2 * math.tan(theta_TF / 2))
            h_ = Ri + dr_plasma_side - Rj_
            h__ = Rj_ - Rj__
            A_ = S_tot_WP - (dx_WP * h_ + dx_WP_min1 * h__)
            h___ = A_ / dx_WP_min2
            Rj = Rj__ - h___

    n_turns = math.ceil(dx_WP / Cond_w)
    n_layers = math.ceil(n_spire / n_turns)
    Ke_cavo_tor = [
        2 * E_jckt * JT / Cond_w
        + 2 * tins * E_tins / Cond_w
        + (
            1 / (E_cbl * cable_w / cable_w)
            + 2 / (E_jckt * JT / Cond_w)
            + 2 / (E_tins * tins / Cond_w)
        )
        ** -1
        for _ in range(n_layers)
    ]
    Ke_WP_tor = sum([(n_turns / k) ** -1 for k in Ke_cavo_tor])
    K_ps_tor = E_case * dr_plasma_side / (2 * Ri * math.tan(theta_TF / 2))
    K_lat_tor = (2 / (E_case * (Ri - Rj) / (dx_case * 0.1))) ** -1
    K_vault_tor = E_case * (Rj - Rk) / (Rk * 2 * math.tan(theta_TF / 2))
    Ke_case_tor = K_ps_tor + (1 / Ke_WP_tor + 1 / K_lat_tor) ** -1 + K_vault_tor
    dcr_vault_tor = K_vault_tor / Ke_case_tor
    S_case = (
        2 * (math.tan(theta_TF / 2) + math.tan(theta_TF / 2)) * (Ri + Rk) * (Ri - Rk) / 2
    )
    eqn = S_amm == +2 / (1 - Rk**2 / Rj**2) * Pm * dcr_vault_tor + FZ / (
        S_steel_WP + S_case - S_tot_WP
    )
    Rk = min([r for r in vpasolve(eqn, Rk) if r > 0]) * 0.95
    radial_build = Ri - Rk
    J_eng = NI / S_tot_WP * 1e-6

    if Rj < 0 or Rk < 0 or Rj_ < Rj:
        print("not feasible!")
        return

    if flag_plot_tf == 1:
        plt.figure(figsize=(8, 8))
        dx_case = 2 * Ri * math.tan(math.pi / n_TF)  # Case width
        dx_case_l = 2 * Rk * math.tan(math.pi / n_TF)  # Case low part width
        plt.fill(
            [-dx_case_l / 2, -dx_case / 2, dx_case / 2, dx_case_l / 2, -dx_case_l / 2],
            [Rk, Ri, Ri, Rk, Rk],
            color="0.7",
        )
        plt.gca().set_aspect("equal")
        TitleName = "TF coil Winding Pack"
        plt.xlabel("[m]", fontsize=20)
        plt.ylabel("[m]", fontsize=20)
        plt.ylim([Rk * 0.8, Ri * 1.2])
        plt.xlim([-dx_case * 2, dx_case * 2])
        dx_ = -dx_case * 1.01
        plt.plot([dx_, dx_], [0, Rk], "--.r", linewidth=1)
        plt.plot([dx_, dx_], [Rk, Rj], "--.g", linewidth=1)
        plt.plot([dx_, dx_], [Rj, Ri], "--.b", linewidth=1)
        plt.plot([-dx_WP / 2, dx_WP / 2], [Ri * 1.03, Ri * 1.03], "--.k", linewidth=1)
        plt.plot(
            [-dx_case / 2, dx_case / 2], [Ri * 1.08, Ri * 1.08], "--.k", linewidth=1
        )
        dx_ = -dx_case

        plt.plot([-dx_, -dx_], [Rj, (Rj + (Ri - Rj))], "--.k", linewidth=1)
        plt.plot([-dx_, -dx_], [Rk, Rj], "--.k", linewidth=1)
        plt.plot([-dx_, -dx_], [(Rj + (Ri - Rj)), Ri], "--.k", linewidth=1)

        plt.text(
            dx_ * 1.03, Rj + (Ri - Rj) / 2, f"WP(h)={round(Ri - Rj, 2)} m", fontsize=12
        )
        plt.text(
            dx_ * 1.03, (Rk + Rj) / 2, f"Nose(h)={round(Rj - Rk, 2)} m", fontsize=12
        )
        plt.text(
            dx_ * 1.03,
            ((Rj + Ri - Rj) + Ri) / 2,
            f"dps(h)={round(dr_plasma_side, 2)} m",
            fontsize=12,
        )

        plt.text(dx_, Rk, f"Rk={round(Rk, 2)} m", fontsize=12)
        plt.text(dx_, Rj, f"Rj={round(Rj, 2)} m", fontsize=12)
        plt.text(dx_, Ri, f"Ri={round(Ri, 2)} m", fontsize=12)
        plt.text(
            0,
            Ri * 1.1,
            f"Case(w)={round(dx_case, 2)} m",
            horizontalalignment="center",
            fontsize=12,
        )
        plt.text(
            0,
            Ri * 1.05,
            f"WP(w)={round(dx_WP, 2)} m",
            horizontalalignment="center",
            fontsize=12,
        )

        dim = [0.85, 0.4, 0.1, 0.5]
        str_ = (
            f"R0={round(R0, 2)} m, B0={round(B0, 2)} T, A={round(A, 2)}, n coil={n_TF}, d={round(d, 2)} m, Jeng={round(J_eng, 1)} Amm-2, "
            f"NI={round(NI * 1e-6, 3)} MA, Bmax={round(BI_TF, 2)} T, discharge time={round(Tau_discharge)} s, Vmax={round(V_MAX)} V, "
            f"Samm={round(S_amm * 1e-6)} MPa, E TF system={round(E)} MJ"
        )
        plt.gca().add_patch(plt.Rectangle(dim, facecolor="y"))
        plt.gca().annotate(str_, dim, fontsize=14)

        plt.title(TitleName, fontsize=20)
        plt.savefig(TitleName + ".png", dpi=300)

    return Rk, Rj, radial_build, J_eng, dx_case, dx_WP


def Jc_REBCO(B):
    A_tapes = 0.4 * 1  # mm2
    T = 6.2
    Tc = 92.83  # K
    Birr0 = 120  # T B ortogonale
    C = 12510  # A T
    p = 0.5
    q = 1.7
    a = 1.52
    b = 2.33
    Birr = Birr0 * (1 - T / Tc) ** a
    Ic_REBCO = C / B * (Birr / Birr0) ** b * (B / Birr) ** p * (1 - B / Birr) ** q
    Jc = Ic_REBCO / A_tapes
    return Jc


def Ths_TF_Nb3Sn(Iop, Bp, Cus, Cuin, NB3SN, Tau_discharge):
    RRR_seg = 300
    RRR_inSC = 100
    Tc = 6.8  # K
    Tau_delay = 3  # s
    t = np.arange(0, Tau_discharge * 5, Tau_discharge / 100)  # s time
    dt = t[1] - t[0]
    Cu_cross_section = Cus + Cuin  # total Cu cross-section [mm^2]
    Icu = np.zeros(t.shape)
    B = np.zeros(t.shape)
    energy = np.zeros(t.shape)
    Jcu = np.zeros(t.shape)
    f = np.zeros(t.shape)
    deltaT = np.zeros(t.shape[0] + 1)
    T_calc = np.zeros(t.shape[0] + 1)

    for i in range(t.shape[0]):
        if t[i] <= Tau_delay:
            Icu[i] = Iop
            B[i] = Bp
        else:
            Icu[i] = Iop * np.exp(-((t[i] - Tau_delay) / Tau_discharge))
            B[i] = Bp * np.exp(-((t[i] - Tau_delay) / Tau_discharge))
        Jcu[i] = Icu[i] / Cu_cross_section

    deltaT[0] = 0
    T_calc[0] = Tc

    def rho_calculationB_parallel_RRR(
        t,
        T,
        B,
        RRR_inSC,
        RRR_seg,
        Cu_cross_sect_in_SC,
        Cu_cross_sect_seg,
        Cu_cross_section,
    ):
        rhoCu_parallel = (
            Cu_cross_section
            * (rho_calculationB(t, T, B, RRR_seg) * rho_calculationB(t, T, B, RRR_inSC))
            / (
                rho_calculationB(t, T, B, RRR_inSC) * Cu_cross_sect_seg
                + rho_calculationB(t, T, B, RRR_seg) * Cu_cross_sect_in_SC
            )
        )
        return rhoCu_parallel

    def rho_calculationB(t, T, B, RRR):
        rhoCu = np.zeros(t.shape)
        rho1 = (1.171 * (10**-17) * (T**4.49)) / (
            1 + (4.5 * (10**-7) * (T**3.35) * (np.exp(-((50 / T) ** 6.428))))
        )
        rho2 = (
            (1.69 * (10**-8) / RRR)
            + rho1
            + 0.4531 * ((1.69 * (10**-8) * rho1) / (RRR * rho1 + 1.69 * (10**-8)))
        )

        for j in range(T.shape[0]):
            for k in range(t.shape[0]):
                A = np.log10(1.553 * (10**-8) * B[k] / rho2[j])
                a = (
                    -2.662
                    + 0.3168 * A
                    + 0.6229 * (A**2)
                    - 0.1839 * (A**3)
                    + 0.01827 * (A**4)
                )
                rhoCu[k, j] = rho2[j] * (1 + (10**a))

        return rhoCu

    def Cu_spec_heat_calculationB(T, Cu_cross_section):
        density = 8960  # Kg/m^3
        cp300 = 3.454e6  # J/K/m^3 known data point at 300K
        gamma = 0.011  # J/K^2/Kg
        beta = 0.0011  # J/K^4/Kg
        c_plow = (beta * (T**3)) + (gamma * T)  # J/K-Kg low temperature range
        Cp_Cu = (
            1 / ((1 / cp300) + (1 / (c_plow * density)))
        ) * Cu_cross_section  # J/K/m^3 volumetric specific heat for the whole temperature range
        return Cp_Cu

    def Nb3Sn_spec_heat_calculationB(T, NB3SN):
        Cp_Nb3Sn = np.zeros(T.shape)
        gamma_Nb = 0.1  # J/K^2/Kg
        beta_Nb = 0.001  # J/K^4/Kg
        density_Nb = 8040  # Kg/m^3
        Cp300_Nb = 210  # J/K/Kg

        Cp_low_NC = (beta_Nb * (T**3)) + (gamma_Nb * T)  # J/K/Kg NORMAL

        for kk in range(T.shape[0]):
            Cp_Nb3Sn[kk] = 1 / ((1 / Cp300_Nb) + (1 / Cp_low_NC[kk]))

        Cp_Nb3Sn = Cp_Nb3Sn * density_Nb * NB3SN
        return Cp_Nb3Sn

    for t_ind in range(t.shape[0]):
        energy[t_ind] = (
            (Jcu[t_ind] ** 2)
            * rho_calculationB_parallel_RRR(
                t[t_ind],
                T_calc[t_ind],
                B[t_ind],
                RRR_inSC,
                RRR_seg,
                Cuin,
                Cus,
                Cu_cross_section,
            )
            * Cu_cross_section
            * dt
        )

        f[t_ind] = Cu_spec_heat_calculationB(
            T_calc[t_ind], Cu_cross_section
        ) + Nb3Sn_spec_heat_calculationB(T_calc[t_ind], NB3SN)

        deltaT[t_ind + 1] = energy[t_ind] / f[t_ind]
        T_calc[t_ind + 1] = T_calc[t_ind] + deltaT[t_ind + 1]

    Ths = round(np.max(T_calc))
    return Ths


def Ic_Nb3Sn_WST(B, d_fili):
    # Constants
    strand_d = d_fili
    CunonCu = 1
    strand_A = (
        math.pi * strand_d**2 / (4 * (1 + CunonCu))
    )  # Area of superconductor in the strand
    c_ = 1.0
    Ca1 = 50.06  # Deviatoric strain
    Ca2 = 0.00  # Deviatoric strain
    eps_0a = 0.00312  # Hydrostatic strain
    eps_m = -0.00059  # Thermal pre-strain
    Bc20max = 33.24  # Maximum upper critical field [T]
    Tc0max = 16.34  # Maximum critical temperature[K]
    C = 83075 * strand_A  # Pre-constant [AT]
    p = 0.593
    q = 2.156

    # Input
    T = 4.2 + 1.5  # Temperature
    # B = 14  # Field
    int_eps = -0.36 / 100  # Intrinsic strain R&W -0.36, W&R -0.55
    # app_eps = int_eps - eps_m  # Applied strain

    # Fit functions
    eps_sh = Ca2 * eps_0a / (math.sqrt(Ca1**2 - Ca2**2))
    s_eps = 1 + (
        Ca1
        * (
            math.sqrt(eps_sh**2 + eps_0a**2)
            - math.sqrt((int_eps - eps_sh) ** 2 + eps_0a**2)
        )
        - Ca2 * int_eps
    ) / (1 - Ca1 * eps_0a)
    Bc0_eps = Bc20max * s_eps
    Tc0_eps = Tc0max * (s_eps) ** (1 / 3)
    t = T / Tc0_eps
    BcT_eps = Bc0_eps * (1 - t ** (1.52))
    TcB_eps = Tc0max * (s_eps) ** (1 / 3) * (1 - B / Bc0_eps) ** (1 / 1.52)
    b = B / BcT_eps
    hT = (1 - t ** (1.52)) * (1 - t**2)
    fPb = b**p * (1 - b) ** q

    # Critical values
    Ic_A = c_ * (C / B) * s_eps * fPb * hT
    Jc_sc = Ic_A / strand_A
    Je_strand = Ic_A / (math.pi * strand_d**2 / 4)

    return Ic_A, Jc_sc


if __name__ == "__main__":
    # design point
    B0 = 6  # field on the axis
    R0 = 2.60  # major radius
    A = 4.5  # aspect ratio
    n_TF = 18  # TF number
    d = 0.9  # SB+VV+gap
    rho = 0.006  # ripple
    a = R0 / A  # minor radius
    Ri = R0 - a - d  # inner leg outer radius
    Re = (R0 + a) * (1 + 1 / rho) ** (1 / n_TF)  # outer leg inner radius

    # Find TF thickness
    Rk, Rj, radial_build, J_eng, dx_case, dx_WP = find_TF_thickness(
        n_TF, Ri, Re, R0, B0, A, d
    )

    # Print the results
    print("Rk:", Rk)
    print("Rj:", Rj)
    print("Radial build:", radial_build)
    print("J_eng:", J_eng)
    print("dx_case:", dx_case)
    print("dx_WP:", dx_WP)
