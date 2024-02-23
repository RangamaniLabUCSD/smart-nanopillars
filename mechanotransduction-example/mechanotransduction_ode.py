import numpy as np
from scipy.integrate import odeint

def mechanotransduction_ode_calc(timeSpan, stiffness, geoParam):
    # # YAP/TAZ and MRTF mechanotransduction model, originally implemented in
    # # VCell
    # #
    # # input:
    # #     timeSpan is a vector of start and stop times (e.g. timeSpan = [0 10.0])
    # #     stiffness specifies the stiffness in kPa
    # #     geoParam is a four-element vector [cytoVol, nucVol, PMArea, NMArea]
    # #
    # # output:
    # #     sol gives the solution computed by odeint

    # #
    # # Initial Conditions
    # #
    yinit = [
        0.2,		# yinit(1) is the initial condition for 'CofilinP'
        0.7,		# yinit(2) is the initial condition for 'Fak'
        0.8,		# yinit(3) is the initial condition for 'mDia'
        0.0,		# yinit(4) is the initial condition for 'LaminA'
        17.9,		# yinit(5) is the initial condition for 'Fcyto'
        33.6,		# yinit(6) is the initial condition for 'RhoAGTP_MEM'
        0.0,		# yinit(7) is the initial condition for 'mDiaA'
        0.0,		# yinit(8) is the initial condition for 'NPCA'
        482.4,		# yinit(9) is the initial condition for 'Gactin'
        6.5,		# yinit(10) is the initial condition for 'NPC'
        0.0,		# yinit(11) is the initial condition for 'ROCKA'
        3.5,		# yinit(12) is the initial condition for 'Myo'
        1.8,		# yinit(13) is the initial condition for 'CofilinNP'
        3500.0,		# yinit(14) is the initial condition for 'LaminAp'
        0.7,		# yinit(15) is the initial condition for 'YAPTAZnuc'
        0.3,		# yinit(16) is the initial condition for 'Fakp'
        0.2,		# yinit(17) is the initial condition for 'YAPTAZP'
        0.7,		# yinit(18) is the initial condition for 'YAPTAZN'
        1.0,		# yinit(19) is the initial condition for 'RhoAGDP'
        1.9,		# yinit(20) is the initial condition for 'LIMK'
        1.5,		# yinit(21) is the initial condition for 'MyoA'
        1.0,		# yinit(22) is the initial condition for 'ROCK'
        geoParam[4]/geoParam[2],		# yinit(23) is the initial condition for 'Positionboolean', scales activation at surface
        0.1,		# yinit(24) is the initial condition for 'LIMKA'
        .65,        # yinit(25) is the initial condition for 'MRTF'
        .301      # yinit(26) is the initial condition for 'MRTFnuc'
    ]

    # #
    # # Default Parameters
    # #   constants are only those "Constants" from the Math Description that are just floating point numbers (no identifiers)
    # #   note: constants of the form "A_init" are really initial conditions and are treated in "yinit"
    # #
    param = [
        0.648,		# param(1) is 'krp'
        0.14,		# param(2) is 'kNC'
        0.4,		# param(3) is 'kra'
        0.7,		# param(4) is 'YAPTAZnuc_init_uM'
        0.015,		# param(5) is 'kf'
        4.0,		# param(6) is 'kmCof'
        4.0,		# param(7) is 'kfc1'
        0.625,		# param(8) is 'kdp'
        0.067,		# param(9) is 'kdmy'
        3.5,		# param(10) is 'Myo_init_uM'
        0.0,		# param(11) is 'mDiaA_init_uM'
        2.0,		# param(12) is 'kdl'
        1.5,		# param(13) is 'MyoA_init_uM'
        0.035,		# param(14) is 'kdf'
        8.7,		# param(15) is 'krNPC'
        0.0,		# param(16) is 'ROCKA_init_uM'
        stiffness,		# param(17) is 'Emol'
        0.34,		# param(18) is 'kcatcof'
        1.8255,		# param(19) is 'SAV'
        0.8,		# param(20) is 'kdrock'
        1.0,		# param(21) is 'RhoAGDP_init_uM'
        0.0168,		# param(22) is 'kfkp'
        geoParam[0], #2300.0,		# param(23) is 'Size_Cyto'
        0.005,		# param(24) is 'kdmdia'
        50.0,		# param(25) is 'alpha'
        1704.7954979572205,		# param(26) is 'Size_ECM'
        0.0,		# param(27) is 'Voltage_PM'
        0.03,		# param(28) is 'kmr'
        77.56,		# param(29) is 'gamma'
        0.002,		# param(30) is 'kmp'
        1.8,		# param(31) is 'CofilinNP_init_uM'
        16.0,		# param(32) is 'k11'
        1.0,		# param(33) is 'netValence_r5'
        1.0,		# param(34) is 'netValence_r4'
        1.0,		# param(35) is 'netValence_r3'
        1.0,		# param(36) is 'netValence_r1'
        0.46,		# param(37) is 'kflaminA'
        120.0,		# param(38) is 'kly'
        0.3,		# param(39) is 'Fakp_init_uM'
        0.07,		# param(40) is 'klr'
        16.0,		# param(41) is 'kll'
        0.0,		# param(42) is 'LaminA_init_molecules_um_2'
        0.165,		# param(43) is 'mDiaB'
        0.0,		# param(44) is 'Voltage_NM'
        33.6,		# param(45) is 'RhoAGTP_MEM_init_molecules_um_2'
        2.8E-7,		# param(46) is 'kfNPC'
        9.64853321E-5,		# param(47) is 'mlabfix_F_nmol_'
        602.2,		# param(48) is 'unitconversionfactor'
        300.0,		# param(49) is 'mlabfix_T_'
        0.8,		# param(50) is 'mDia_init_uM'
        1000.0,		# param(51) is 'K_millivolts_per_volt'
        0.001,		# param(52) is 'Kr_r15'
        0.0,		# param(53) is 'Kr_r12'
        100.0,		# param(54) is 'Clamin'
        36.0,		# param(55) is 'epsilon'
        0.2,		# param(56) is 'YAPTAZP_init_uM'
        17.9,		# param(57) is 'Fcyto_init_uM'
        0.3,		# param(58) is 'ROCKB'
        482.4,		# param(59) is 'Gactin_init_uM'
        geoParam[2], #1260.0,		# param(60) is 'Size_PM'
        0.1,		# param(61) is 'LIMKA_init_uM'
        geoParam[1], #550.0,		# param(62) is 'Size_Nuc'
        10.0,		# param(63) is 'kin2'
        3.141592653589793,		# param(64) is 'mlabfix_PI_'
        9.0E-6,		# param(65) is 'p'
        96485.3321,		# param(66) is 'mlabfix_F_'
        1.0,		# param(67) is 'kinSolo2'
        8314.46261815,		# param(68) is 'mlabfix_R_'
        55.49,		# param(69) is 'tau'
        1.0,		# param(70) is 'kout2'
        1.0,		# param(71) is 'ROCK_init_uM'
        1.0E-9,		# param(72) is 'mlabfix_K_GHK_'
        3.5,		# param(73) is 'kdep'
        geoParam[3], #393.0,		# param(74) is 'Size_NM'
        0.0,		# param(75) is 'Kr_r7'
        0.0,		# param(76) is 'Kr_r6'
        0.0,		# param(77) is 'Kr_r5'
        3.25,		# param(78) is 'C' # this is eventually changed for the full mechanoCircadian model
        0.0,		# param(79) is 'Kr_r4'
        0.0,		# param(80) is 'Kr_r3'
        0.0,		# param(81) is 'Kr_r2'
        0.0,		# param(82) is 'Kr_r1'
        0.0,		# param(83) is 'Kr_r0'
        6.5,		# param(84) is 'NPC_init_molecules_um_2'
        6.02214179E11,		# param(85) is 'mlabfix_N_pmol_'
        7.6E-4,		# param(86) is 'kCY'
        0.2,		# param(87) is 'CofilinP_init_uM'
        3500.0,		# param(88) is 'LaminAp_init_molecules_um_2'
        0.56,		# param(89) is 'kCN'
        1.0,		# param(90) is 'netValence_r16'
        1.0,		# param(91) is 'netValence_r15'
        1.0,		# param(92) is 'netValence_r14'
        1.9,		# param(93) is 'LIMK_init_uM'
        1.0,		# param(94) is 'netValence_r12'
        0.04,		# param(95) is 'kturnover'
        0.7,		# param(96) is 'Fak_init_uM'
        0.379,		# param(97) is 'ksf'
        0.7,		# param(98) is 'YAPTAZN_init_uM'
        5.0,		# param(99) is 'n2'
        2.6,		# param(100) is 'n1'
        0.0,		# param(101) is 'NPCA_init_molecules_um_2'
        geoParam[4]/geoParam[2],		# param(102) is positionBoolean, scales activation at surface
        0.001660538783162726,		# param(103) is 'KMOLE'
    ]

    # ode rate
    def f(y, t, p):
        # State Variables
        CofilinP, Fak, mDia, LaminA, Fcyto, RhoAGTP_MEM, mDiaA, NPCA, Gactin, NPC = y[0:10]
        ROCKA, Myo, CofilinNP, LaminAp, YAPTAZnuc, Fakp, YAPTAZP, YAPTAZN, RhoAGDP, LIMK = y[10:20]
        MyoA, ROCK, Positionboolean, LIMKA, MRTF, MRTFnuc = y[20:26]
        # Constants
        krp, kNC, kra = p[0:3]
        kf, kmCof, kfc1, kdp, kdmy = p[4:9]
        kdl = p[11]
        kdf, krNPC = p[13:15]
        Emol, kcatcof, SAV, kdrock = p[16:20]
        kfkp, Size_Cyto, kdmdia, alpha = p[21:25]
        kmr, gamma, kmp = p[27:30]
        kflaminA = p[36]
        klr = p[39]
        mDiaB = p[42]
        kfNPC = p[45]
        unitconversionfactor = p[47]
        Kr_r15, Kr_r12, Clamin, epsilon = p[51:55]
        ROCKB = p[57]
        Size_PM = p[59]
        Size_Nuc, kin2 = p[61:63]
        propStiff = p[64]
        kinSolo2 = p[66]
        tau, kout2 = p[68:70]
        kdep, Size_NM, Kr_r7, Kr_r6, Kr_r5, C, Kr_r4, Kr_r3, Kr_r2, Kr_r1, Kr_r0 = p[72:83]
        kCY = p[85]
        kCN = p[88]
        kturnover = p[94]
        ksf = p[96]
        n2, n1 = p[98:100]
        # Functions
        Kf_r9 = (kmr * (1.0 + (0.5 * epsilon * (1.0 + np.tanh((20.0 * (ROCKA - ROCKB)))) * ROCKA)))
        Kf_r8 = (klr * (1.0 + (0.5 * tau * (1.0 + np.tanh((20.0 * (ROCKA - ROCKB)))) * ROCKA)))
        Kf_r7 = kdmdia
        Kf_r6 = kdrock
        Kf_r5 = (kmp * RhoAGTP_MEM)
        Kf_r4 = (krp * RhoAGTP_MEM)
        Kf_r3 = kdp
        Kf_r2 = kf
        Kf_r1 = (kfkp * (1.0 + (gamma * (Fakp ** n2))) * unitconversionfactor * SAV)
        Kf_r0 = kdf
        Ecytosol = (propStiff * (Fcyto ** n1))
        Kr_r9 = kdmy
        J_r9 = ((Kf_r9 * Myo) - (Kr_r9 * MyoA))
        Kr_r8 = kdl
        J_r8 = ((Kf_r8 * LIMK) - (Kr_r8 * LIMKA))
        J_r7 = ((Kf_r7 * mDiaA) - (Kr_r7 * mDia))
        J_r6 = ((Kf_r6 * ROCKA) - (Kr_r6 * ROCK))
        J_r5 = ((Kf_r5 * mDia) - (Kr_r5 * mDiaA))
        J_r4 = ((Kf_r4 * ROCK) - (Kr_r4 * ROCKA))
        J_r3 = ((Kf_r3 * RhoAGTP_MEM) - (Kr_r3 * RhoAGDP))
        J_r2 = ((Kf_r2 * Fak) - (Kr_r2 * Fakp))
        J_r1 = ((Kf_r1 * RhoAGDP) - (Kr_r1 * RhoAGTP_MEM))
        J_r0 = ((Kf_r0 * Fakp) - (Kr_r0 * Fak))
        KFlux_NM_Nuc = (Size_NM / Size_Nuc)
        Kr_r16 = kout2
        Kf_r16 = ((kin2 * NPCA) + kinSolo2)
        J_r16 = ((Kf_r16 * YAPTAZN) - (Kr_r16 * YAPTAZnuc))
        Kf_r15 = (kflaminA * Ecytosol / (Clamin + Ecytosol))
        J_r15 = ((Kf_r15 * LaminAp) - (Kr_r15 * LaminA))
        Kr_r14 = krNPC
        Kf_r14 = (kfNPC * MyoA * Fcyto * LaminA)
        J_r14 = ((Kf_r14 * NPC) - (Kr_r14 * NPCA))
        Kr_r13 = kNC
        Kf_r13 = (kCN + (kCY * MyoA * Fcyto))
        J_r13 = ((Kf_r13 * YAPTAZP) - (Kr_r13 * YAPTAZN))
        Kf_r12 = (ksf * Emol * unitconversionfactor * SAV * Positionboolean / (C + Emol))
        J_r12 = ((Kf_r12 * Fak) - (Kr_r12 * Fakp))
        Kr_r11 = (kdep + (kfc1 * CofilinNP))
        Kf_r11 = (kra * (1.0 + (0.5 * alpha * (1.0 + np.tanh((20.0 * (mDiaA - mDiaB)))) * mDiaA)))
        J_r11 = ((Kf_r11 * Gactin) - (Kr_r11 * Fcyto))
        Kr_r10 = (kcatcof * LIMKA / (kmCof + CofilinNP))
        Kf_r10 = kturnover
        J_r10 = ((Kf_r10 * CofilinP) - (Kr_r10 * CofilinNP))
        KFlux_PM_Cyto = (Size_PM / Size_Cyto)
        KFlux_NM_Cyto = (Size_NM / Size_Cyto)
        UnitFactor_uM_um3_molecules_neg_1 = (1000000.0 / 6.02214179E8)
        
        # define MRTF fluxes, MRTFTot should be 3.15e6
        MRTFReleaseConst = 208.4 #43.2495; #112.5; #MRTFReleaseConst
        kinsolo_MRTF = 2.708 #8.4028; #2.6383; #kinsolo_MRTF
        kin2_MRTF = 8.5 #16.5570; #6.2886; #kin2_MRTF
        kout_MRTF = 1.0 #0.1346; #0.2725; #kout_MRTF
        Kr_MRTF = kout_MRTF
        Kf_MRTF = ((kin2_MRTF * NPCA) + kinsolo_MRTF) * (1/(1 + (Gactin / MRTFReleaseConst) ** 2.0))
        J_MRTF = ((Kf_MRTF * MRTF) - (Kr_MRTF * MRTFnuc))

        # Rates
        dydt = [
            - J_r10,   # rate for CofilinP
            ( - (UnitFactor_uM_um3_molecules_neg_1 * KFlux_PM_Cyto * J_r12) + J_r0 - SAV*KFlux_PM_Cyto*J_r2),    # rate for Fak
            ( - (UnitFactor_uM_um3_molecules_neg_1 * KFlux_PM_Cyto * J_r5) + J_r7),    # rate for mDia
            J_r15,    # rate for LaminA
            J_r11,    # rate for Fcyto
            ( - J_r3 + J_r1),    # rate for RhoAGTP_MEM
            ((UnitFactor_uM_um3_molecules_neg_1 * KFlux_PM_Cyto * J_r5) - J_r7),    # rate for mDiaA
            J_r14,    # rate for NPCA
            - J_r11,    # rate for Gactin
            - J_r14,    # rate for NPC
            ((UnitFactor_uM_um3_molecules_neg_1 * KFlux_PM_Cyto * J_r4) - J_r6),    # rate for ROCKA
            - J_r9,    # rate for Myo
            J_r10,    # rate for CofilinNP
            - J_r15,    # rate for LaminAp
            (UnitFactor_uM_um3_molecules_neg_1 * KFlux_NM_Nuc * J_r16),    # rate for YAPTAZnuc
            ((UnitFactor_uM_um3_molecules_neg_1 * KFlux_PM_Cyto * J_r12) - J_r0 + SAV*KFlux_PM_Cyto*J_r2),    # rate for Fakp
            - J_r13,    # rate for YAPTAZP
            (J_r13 - (UnitFactor_uM_um3_molecules_neg_1 * KFlux_NM_Cyto * J_r16)),    # rate for YAPTAZN
            ((UnitFactor_uM_um3_molecules_neg_1 * KFlux_PM_Cyto * J_r3) - (UnitFactor_uM_um3_molecules_neg_1 * KFlux_PM_Cyto * J_r1)),    # rate for RhoAGDP
            - J_r8,    # rate for LIMK
            J_r9,    # rate for MyoA
            ( - (UnitFactor_uM_um3_molecules_neg_1 * KFlux_PM_Cyto * J_r4) + J_r6),    # rate for ROCK
            0.0,    # rate for Positionboolean
            J_r8,    # rate for LIMKA
            -(UnitFactor_uM_um3_molecules_neg_1 * KFlux_NM_Cyto * J_MRTF),    # rate for MRTF
            (UnitFactor_uM_um3_molecules_neg_1 * KFlux_NM_Nuc * J_MRTF),    # rate for MRTFnuc
        ]
        return dydt

    #
    # invoke the integrator
    #
    t = np.linspace(timeSpan[0], timeSpan[1], 1000)
    sol = odeint(f, yinit, t, args=(param,))
    return t, sol