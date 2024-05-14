import dolfin as d
import sympy as sym
import numpy as np

from smart import config, mesh, model, visualization
from smart.units import unit
from smart.model_assembly import (
    Compartment,
    Parameter,
    Reaction,
    Species,
    CompartmentContainer,
    sbmodel_from_locals,
)


def load_model(mesh_file):
    # Aliases - base units
    uM = unit.uM
    um = unit.um
    molecule = unit.molecule
    sec = unit.sec
    dimensionless = unit.dimensionless
    # Aliases - units used in model
    D_unit = um**2 / sec
    flux_unit = uM * um / sec
    vol_unit = uM
    surf_unit = molecule / um**2
    # electrical units
    voltage_unit = unit.millivolt
    current_unit = unit.picoampere
    conductance_unit = unit.picosiemens

    # ## Load in mesh
    #
    # We load a realistic spine mesh, including the spine with a postsynaptic density (PSD) and spine apparatus (SA). Note that we store the location of the PSD from the original mesh (facets labeled with the value 11) to the mesh function `facet_markers_orig`, but otherwise, rewrite the mesh so that all of the PM (including the PSD) is treated as a single domain.

    # +

    receptors_as_param = False

    parent_mesh = mesh.ParentMesh(
        mesh_filename=mesh_file,
        mesh_filetype="hdf5",
        name="parent_mesh",
        extra_keys=["subdomain0_2", "subdomain1_3"],
    )

    subdomain0 = parent_mesh.subdomains[0]
    subdomain1 = parent_mesh.subdomains[1]
    ds = d.Measure("ds", domain=parent_mesh.dolfin_mesh, subdomain_data=subdomain0)
    A_PSD = d.assemble_mixed(1.0 * ds(11))
    visualization.plot_dolfin_mesh(parent_mesh.dolfin_mesh, subdomain1)
    # -

    # # Model generation
    #
    # For each step of model generation, refer to SMART Example 3 or API documentation for further details.
    #
    # We first define compartments and the compartment container. Note that we can specify nonadjacency for surfaces in the model, which is not required, but can speed up the solution process.

    # +
    Cyto = Compartment("Cyto", 3, um, 1)
    PM = Compartment("PM", 2, um, 10)
    SA = Compartment("SA", 3, um, 2)
    SAm = Compartment("SAm", 2, um, 12)
    PM.specify_nonadjacency(["SAm", "SA"])
    SAm.specify_nonadjacency(["PM"])

    cc = CompartmentContainer()
    cc.add([Cyto, PM, SA, SAm])
    # -

    # Define species and place them in a species container. Note that `NMDAR` and `VSCC` are stationary PM surface variables, effectively just serving the role restricting NMDAR calcium influx to the PSD and VSCC influx to the spine (not the dendritic shaft). Additionally, unlike the original model, we consider changes in CaSA rather than treating it as a constant. To account for buffering of the calcium in the SR, we use an effective diffusion coefficient that is much slower than free calcium diffusion and we also rescale all fluxes in and out of the SA (implemented after reaction definitions below). As a rough approximation, we assume a low-affinity buffer ($K_D = 2000 \mu M$) in the SA lumen at a total concentration of $B_{SA,tot} =$ 10 mM. Assuming the buffer has a diffusion coefficient $D_B = 10 \mu m^2/s$, the SA calcium will have an effective diffusion coefficient of:
    # $$
    # D_{CaSA,eff} = \frac{K_D}{K_D + B_{SA,tot}}(D_{Ca,free} + \frac{B_{SA,tot}}{K_D} D_B) = 45 \mu m^2/s
    # $$
    #
    # Similarly, changes in CaSA due to fluxes in and out of the SA need to be multiplied by the factor:
    # $$
    # \xi_{SA} = \frac{1}{1 + B_{SA,tot}/K_D} = 1/6
    # $$

    # +
    Ca = Species("Ca", 0.1, vol_unit, 220.0, D_unit, "Cyto")
    n_PMr = 0.1011  # vol to surf area ratio for a realistic dendritic spine
    # note that NMDAR initial condition is overwritten later to localize to PSD
    if receptors_as_param:
        NMDAR = Parameter(
            "NMDAR", 1.0, dimensionless
        )  # specify parameter to localize NMDAR calcium influx to PSD
        VSCC_zThresh = -0.25  # 0.3 #-0.25 for single spine, 0.3 for 2 spine
        VSCC_loc = f"(1 + sign(z - {VSCC_zThresh}))/2"
        VSCC = Parameter.from_expression(
            "VSCC", VSCC_loc, dimensionless
        )  # specify parameter to localize VSCC calcium influx to spine body and neck
    else:
        NMDAR = Species(
            "NMDAR",
            1.0,
            dimensionless,
            0.0,
            D_unit,
            "PM",
        )  # specify species to localize NMDAR calcium influx to PSD
        NMDAR.restrict_to_subdomain(subdomain0, 11)
        VSCC_zThresh = -0.25  # 0.3 #-0.25 for single spine, 0.3 for 2 spine
        VSCC_loc = f"(1 + sign(z - {VSCC_zThresh}))/2"
        VSCC = Species(
            "VSCC", VSCC_loc, dimensionless, 0.0, D_unit, "PM"
        )  # specify species to localize VSCC calcium influx to spine body and neck

    Bf = Species("Bf", 78.7 * n_PMr, vol_unit * um, 0.0, D_unit, "PM")
    Bm = Species("Bm", 20.0, vol_unit, 20.0, D_unit, "Cyto")
    CaSA = Species(
        "CaSA", 60.0, vol_unit, 45.0, D_unit, "SA"
    )  # effective D due to buffering
    # -

    # Define parameters and reactions at the plasma membrane:
    # * a1: influx of calcium through N-methyl-D-aspartate receptors (NMDARs), localized to the PSD
    # * a2: calcium entry through voltage-sensitive calcium channels (VSCCs) throughout the PM
    # * a3: calcium efflux through PMCA (all PM)
    # * a4: calcium efflux through NCX (all PM)
    # * a5: calcium binding to immobilized buffers
    #
    # Calcium entry through NMDARs and VSCCs are given by time-dependent functions. Each depends on the voltage over time, which is specified to match the expected dynamics due to the back-propagating action potential (BPAP) and the excitatory postsynaptic potential (EPSP):
    #
    # $$
    # V_m (t) = V_{rest} + BPAP(t) + EPSP(t)\\
    # = V_{rest} + [BPAP]_{max} \left( I_{bsf} e^{-(t-t_{delay,bp})/t_{bsf}} + I_{bss} e^{-(t-t_{delay,bp})/t_{bss}}\right) + s_{term} \left( e^{-(t-t_{delay})/t_{ep1}} - e^{-(t-t_{delay})/t_{ep2}}\right)
    # $$
    #
    # For an overview of the additional equations, refer to the original publication [Bell et al 2019, Journal of General Physiology](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6683673/).

    # +
    # Both NMDAR and VSCC fluxes depend on the voltage over time
    Vrest_expr = -65
    Vrest = Parameter("Vrest", Vrest_expr, voltage_unit)
    bpmax = 38
    Ibsf, tbsf, Ibss, tbss = 0.75, 0.003, 0.25, 0.025
    tdelaybp, tdelay = 0.002, 0.0
    sterm = 25
    tep1, tep2 = 0.05, 0.005
    t = sym.symbols("t")
    BPAP = (
        bpmax
        * (
            Ibsf * sym.exp(-(t - tdelaybp) / tbsf)
            + Ibss * sym.exp(-(t - tdelaybp) / tbss)
        )
        * (1 + sym.sign(t))
        / 2
    )
    EPSP = (
        sterm
        * (sym.exp(-(t - tdelay) / tep1) - sym.exp(-(t - tdelay) / tep2))
        * (1 + sym.sign(t))
        / 2
    )
    Vm_expr = Vrest_expr + BPAP + EPSP
    # no preintegration for Vm (not a flux)
    Vm = Parameter.from_expression(
        "Vm",
        Vm_expr,
        voltage_unit,
        use_preintegration=False,
    )
    # Define known constants
    N_A = 6.022e23  # molecules per mole
    F = 96485.332  # Faraday's constant (Coulombs per mole)
    Q = 1.602e-19  # Coulombs per elementary charge

    n_PMr = Parameter("n_PMr", 0.1011, um)
    # NMDAR calcium influx
    Vrev_expr = 90
    Vrev = Parameter("Vrev", Vrev_expr, voltage_unit)
    P0 = 0.5
    CaEC = 2  # mM
    h = 11.3  # pS/mM
    G0 = 65.6  # pS
    r = 0.5
    ginf = 15.2  # pS
    convert_factor = 1e-15  # A/mV per pS
    zeta_i = (G0 + r * CaEC * h) / (
        1 + r * CaEC * h / ginf
    )  # single channel conductance in pS
    G_NMDARVal = convert_factor * zeta_i / (2 * Q)
    G_NMDAR = Parameter("G_NMDAR", G_NMDARVal, molecule / (voltage_unit * sec))
    If, tau_f, Is, tau_s = 0.5, 0.05, 0.5, 0.2
    Km = 0.092  # (1/mV)
    Mg, MgScale = 1, 3.57  # mM
    B_V = 1 / (1 + sym.exp(-Km * Vm_expr) * Mg / MgScale)
    gamma_i_scale = (
        P0
        * (If * sym.exp(-t / tau_f) + Is * sym.exp(-t / tau_s))
        * B_V
        * (1 + sym.sign(t))
        / 2
    )
    beta_NMDAR = 85
    J0_NMDAR_expr = -gamma_i_scale / (
        beta_NMDAR * A_PSD
    )  # negative sign because of inward current
    J0_NMDAR = Parameter.from_expression(
        "J0_NMDAR",
        J0_NMDAR_expr,
        1 / um**2,
        use_preintegration=True,
        numerical_int=True,
    )
    if receptors_as_param:
        a1 = Reaction(
            "a1",
            [],
            ["Ca"],
            species_map={},
            param_map={
                "J0": "J0_NMDAR",
                "G_NMDAR": "G_NMDAR",
                "Vm": "Vm",
                "Vrev": "Vrev",
                "NMDAR": "NMDAR",
            },
            eqn_f_str="J0*NMDAR*G_NMDAR*(Vm - Vrev)",
            explicit_restriction_to_domain="PM",
        )
        a1.restrict_to_subdomain(subdomain0, 11)
    else:
        a1 = Reaction(
            "a1",
            [],
            ["Ca"],
            species_map={"NMDAR": "NMDAR"},
            param_map={
                "J0": "J0_NMDAR",
                "G_NMDAR": "G_NMDAR",
                "Vm": "Vm",
                "Vrev": "Vrev",
            },
            eqn_f_str="J0*NMDAR*G_NMDAR*(Vm - Vrev)",
            explicit_restriction_to_domain="PM",
        )
    # VSCC calcium influx
    gamma = 3.72  # pS
    k_Ca = (
        -convert_factor
        * gamma
        * Vm_expr
        * N_A
        * (0.393 - sym.exp(-Vm_expr / 80.36))
        / (2 * F * (1 - sym.exp(Vm_expr / 80.36)))
    )  # negative for inward current
    alpha4, beta4 = 34700, 3680
    VSCC_biexp = (sym.exp(-alpha4 * t) - sym.exp(-beta4 * t)) * (1 + sym.sign(t)) / 2
    VSCCNum = 2  # molecules/um^2
    J_VSCC = Parameter.from_expression(
        "J_VSCC",
        VSCCNum * k_Ca * VSCC_biexp,
        molecule / (um**2 * sec),
        use_preintegration=True,
        numerical_int=True,
    )
    if receptors_as_param:
        a2 = Reaction(
            "a2",
            [],
            ["Ca"],
            species_map={},
            param_map={"J": "J_VSCC", "VSCC": "VSCC"},
            eqn_f_str="J*VSCC",
            explicit_restriction_to_domain="PM",
        )
    else:
        a2 = Reaction(
            "a2",
            [],
            ["Ca"],
            species_map={"VSCC": "VSCC"},
            param_map={"J": "J_VSCC"},
            eqn_f_str="J*VSCC",
            explicit_restriction_to_domain="PM",
        )
    # PMCA
    Prtote = Parameter("Prtote", 191, vol_unit)
    Kme = Parameter("Kme", 2.43, vol_unit)
    Prtotex = Parameter("Prtotex", 8.77, vol_unit)
    Kmx = Parameter("Kmx", 0.139, vol_unit)
    Vmax_lr23 = Parameter("Vmax_lr23", 0.113, vol_unit / sec)
    Km_lr23 = Parameter("Km_lr23", 0.442, vol_unit)
    Vmax_hr23 = Parameter("Vmax_hr23", 0.59, vol_unit / sec)
    Km_hr23 = Parameter("Km_hr23", 0.442, vol_unit)
    beta_PMCA = 100
    beta_i_str = (
        "(1 + Prtote*Kme/(Kme+c)**2 + Prtotex*Kmx/(Kmx+c)**2)**(-1)"  # buffering term
    )
    PMCA_str = "Vmax_lr23*c**2/(Km_lr23**2 + c**2) + Vmax_hr23*c**5/(Km_hr23**5 + c**5)"
    a3 = Reaction(
        "a3",
        ["Ca"],
        [],
        {
            "Prtote": "Prtote",
            "Prtotex": "Prtotex",
            "Kme": "Kme",
            "Kmx": "Kmx",
            "Vmax_lr23": "Vmax_lr23",
            "Km_lr23": "Km_lr23",
            "Vmax_hr23": "Vmax_hr23",
            "Km_hr23": "Km_hr23",
            "n_PMr": "n_PMr",
        },
        {"c": "Ca"},
        eqn_f_str=f"{beta_PMCA}*({beta_i_str})*({PMCA_str})*n_PMr",
        explicit_restriction_to_domain="PM",
    )
    # NCX
    Vmax_r22 = Parameter("Vmax_r22", 0.1, vol_unit / sec)
    Km_r22 = Parameter("Km_r22", 1, vol_unit)  # uM
    beta_NCX = 1000
    NCX_str = "Vmax_r22*c/(Km_r22 + c)"
    a4 = Reaction(
        "a4",
        ["Ca"],
        [],
        {
            "Prtote": "Prtote",
            "Prtotex": "Prtotex",
            "Kme": "Kme",
            "Kmx": "Kmx",
            "Vmax_r22": "Vmax_r22",
            "Km_r22": "Km_r22",
            "n_PMr": "n_PMr",
        },
        {"c": "Ca"},
        eqn_f_str=f"{beta_NCX}*({beta_i_str})*({NCX_str})*n_PMr",
        explicit_restriction_to_domain="PM",
    )
    # Immobilized buffers
    kBf_on = Parameter("kBf_on", 1, 1 / (uM * sec))
    kBf_off = Parameter("kBf_off", 2, 1 / sec)
    Bf_tot = Parameter("Bf_tot", 78.7 * n_PMr.value, vol_unit * um)
    a5 = Reaction(
        "a5",
        ["Ca", "Bf"],
        [],
        {"kon": "kBf_on", "koff": "kBf_off", "Bf_tot": "Bf_tot"},
        eqn_f_str="kon*Ca*Bf - koff*(Bf_tot - Bf)",
        explicit_restriction_to_domain="PM",
    )

    # Now we define the cytosolic reactions. Here, there is only one reaction: mobile buffer binding calcium (b1). Note that because we assume that the buffering protein and the buffering protein bound to calcium have the same diffusion coefficient, we know that the total amount of buffering protein does not change over time or space, and we can write $[CaB_m] = B_{m,tot} - B_m$

    # calcium buffering in the cytosol
    kBm_on = Parameter("kBm_on", 1, 1 / (uM * sec))
    kBm_off = Parameter("kBm_off", 1, 1 / sec)
    Bm_tot = Parameter("Bm_tot", 20, vol_unit)
    b1 = Reaction(
        "b1",
        ["Ca", "Bm"],
        [],
        param_map={"kon": "kBm_on", "koff": "kBm_off", "Bm_tot": "Bm_tot"},
        eqn_f_str="kon*Ca*Bm - koff*(Bm_tot - Bm)",
    )

    # Finally, we define reactions associated with the spine apparatus:
    # * c1: calcium pumping into the SA through SERCA
    # * c2: calcium leak out of the SA

    # +
    # SERCA flux
    n_SAr = Parameter("n_SAr", 0.0113, um)
    Vmax_r19 = Parameter("Vmax_r19", 114, vol_unit / sec)
    KP_r19 = Parameter("KP_r19", 0.2, vol_unit)
    beta_SERCA = 1000
    VmaxSERCA_str = "Vmax_r19*c**2/(KP_r19**2 + c**2)"
    c1 = Reaction(
        "c1",
        ["Ca"],
        ["CaSA"],
        {
            "Prtote": "Prtote",
            "Prtotex": "Prtotex",
            "Kme": "Kme",
            "Kmx": "Kmx",
            "Vmax_r19": "Vmax_r19",
            "KP_r19": "KP_r19",
            "n_SAr": "n_SAr",
        },
        {"c": "Ca"},
        eqn_f_str=f"{beta_SERCA}*({beta_i_str})*({VmaxSERCA_str})*n_SAr",
        explicit_restriction_to_domain="SAm",
    )
    # calcium leak out of the SA
    k_leak = Parameter("k_leak", 0.1608, 1 / sec)
    c2 = Reaction(
        "c2",
        ["CaSA"],
        ["Ca"],
        {"k_leak": "k_leak", "n_SAr": "n_SAr"},
        {"c": "Ca", "cSA": "CaSA"},
        eqn_f_str="k_leak*(cSA - c)*n_SAr",
        explicit_restriction_to_domain="SAm",
    )

    xi = 1 / 6  # scaling factor to account for rapid buffering in SA (see above)
    for c in [c1, c2]:
        c.flux_scaling = {"CaSA": xi}
        c.__post_init__()
    # -

    # Now we add all parameters and reactions to their SMART containers.

    pc, sc, cc, rc = sbmodel_from_locals(locals().values())

    # Initialize model and solver.

    # +
    configCur = config.Config()
    configCur.flags.update({"allow_unused_components": True})
    model_cur = model.Model(pc, sc, cc, rc, configCur, parent_mesh)
    configCur.solver.update(
        {
            "final_t": 0.025,
            "initial_dt": 0.0002,
            "time_precision": 8,
            "use_snes": True,
        }
    )
    model_cur.initialize(initialize_solver=True)
    return model_cur
