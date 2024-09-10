# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Mechanotransduction model in a mammalian cell
#
# Here, we implement the model originally presented in [Sun et al. 2016, Biophysical Journal] and then expanded by [Scott et al. 2021, PNAS]. The goal is to describe the nuclear translocation of the mechanosensitive transcriptional regulators YAP/TAZ, given some mechanical input from the substrate.
#
# The geometry in this model is divided into 4 domains - two volumes and two surfaces:
# - plasma membrane (PM)
# - Cytosol (Cyto)
# - Nuclear membrane (NM)
# - Interior of the nucleus (Nuc)
#
# This model has 19 species after accounting for mass conservation in certain cases. Note that this is only possible in those cases where:
# 1) Both forms of the species live in the same compartment.
# 2) Both forms of the species have the same diffusion coefficient.
#
# In the cases of RhoA, YAPTAZ, and MRTF, the first criterion is not satisfied, whereas for G-Actin/F-Actin, the diffusion coefficients differ considerably, violating condition 2.
#
# In summary, the PM species are:
# * Emod: a stand-in species denoting the Young's modulus of the surface in contact with the cell membrane.
# * RhoA_GTP: activated form of RhoA
#
# The cytosolic species include:
# * pFAK: phosphorylated FAK (unphosphorylated accounted for by mass conservation)
# * RhoA_GDP: inactive form of RhoA
# * ROCK_A: activated Rho-associated protein kinase
# * mDia_A: activated version of the formin mDia
# * Myo_A: activated myosin (associated with stress fiber formation)
# * LIMK_A: activated LIM kinase
# * Cofilin_NP: nonphosphorylated (activated) cofilin
# * FActin: polymerized form of actin
# * GActin: monomeric form of actin
# * YAPTAZ: dephosphorylated form of YAP/TAZ in the cytosol
# * YAPTAZ_phos: phosphorylated form of YAP/TAZ (preferentially remains cytosolic)
#
# The nuclear membrane species include:
# * LaminA: dephosphorylated form of Lamin A, a structural componet of the nuclear lamina
# * NPC_A: activated (open) nuclear pore complexes
#
# Finally, the nucleus contains:
# * YAPTAZ_nuc: concentration of YAP/TAZ in the nucleus

# +
import dolfin as d
import sympy as sym
import numpy as np
import pathlib
import logging
import argparse
import matplotlib.pyplot as plt


from smart import config, mesh, model, mesh_tools
from smart.units import unit
from smart.model_assembly import (
    Compartment,
    Parameter,
    Reaction,
    Species,
    sbmodel_from_locals,
)
import sys

from mech_parser_args import add_mechanotransduction_arguments, shape2symfraction, Shape
here = pathlib.Path.cwd()
sys.path.insert(0, (here / ".." / "scripts").as_posix())
import runner as main_run

smart_logger = logging.getLogger("smart")
smart_logger.setLevel(logging.DEBUG)
logger = logging.getLogger("mechanotransduction")
logger.setLevel(logging.INFO)
logger.info("Starting mechanotransduction example")
# -

# If running a series of tests, we use `argparse` to read in the test conditions. Otherwise, we load default parameters to the `args` dict.

# +
# outerExpr13 = "(1 - z**4/(2000+z**4)) * (r**2 + z**2) + 0.4*(r**2 + (z+9.72)**2)*z**4 / (15 + z**4) - 169"
# innerExpr13 = "(r/5.3)**2 + ((z-9.6/2)/2.4)**2 - 1"

sys.path.insert(0, "/root/shared/gitrepos/smart-nanopillars/utils")
import spread_cell_mesh_generation as mesh_gen
parser = argparse.ArgumentParser()
add_mechanotransduction_arguments(parser)
try: # run as python script on cluster
    args = vars(parser.parse_args())
except: # not a run on the cluster
    # mesh_sphere, mf2, mf3 = mesh_tools.create_ellipsoids(outerRad = [6,6,6], innerRad = [3,3,3],
    #                                               hEdge = 1.0, hInnerEdge = 1.0)
    # coords = mesh_sphere.coordinates()
    # for i in range(len(coords)):
    #     coords[i][2] += 15
    # mesh_tools.write_mesh(mesh_sphere, mf2, mf3, mesh_folder / "spreadCell_mesh.h5")
    # main_run.pre_preprocess_mech_mesh(
    #     mesh_folder=mesh_folder, shape = 'circle',
    #     contact_rad = 15.5,
    #     hEdge=0.6, hInnerEdge=0.6, dry_run = False,
    #     num_refinements=0,
    #     full_3d=False,
    #     nanopillar_radius=0.0,
    #     nanopillar_height=0.0,
    #     nanopillar_spacing=0.0,
    #     nuc_compression=0.0)
    # mesh_folder = pathlib.Path("/root/shared/gitrepos/smart-comp-sci-data/meshes/nanopillars/nanopillars_h1.0_p2.5_r0.1_cellRad18.52")
    # mesh_folder = pathlib.Path("/root/shared/gitrepos/smart-comp-sci-data/meshes/nanopillars/nanopillars_indent1.0_coarser")
    # mesh_folder = pathlib.Path("/root/shared/gitrepos/smart-comp-sci-data/meshes/nanopillars/nanopillars_h0.0_p0.0_r0.0_cellRad18.08")
    # results_folder = pathlib.Path("results_noNPPore")
    args = {}
    args["time_step"] = 0.01
    args["e_val"] = 7e7
    args["z_cutoff"] = 1e-4
    args["axisymmetric"] = False
    args["well_mixed"] = False
    args["curv_sens"] = 5
    args["reaction_rate_on_np"] = 1
    args["npc_slope"] = 0.0
    args["u0_npc"] = 5.0
    args["nuc_compression"] = 1.4
    mesh_folder = pathlib.Path(f"/root/shared/gitrepos/smart-comp-sci-data/meshes/nanopillars_new/nanopillars_indent{args['nuc_compression']}")
    args["mesh_folder"] = mesh_folder
    results_folder = pathlib.Path("nanopillars_indent1.4_u0_5.0_fixed")#f"/root/scratch/nanopillar-sims/results_nanopillars_indentation_newStretch/nanopillars_indent{args['nuc_compression']}_u0_{args['u0_npc']}")
    args["outdir"] = results_folder

hEdge = 0.8
hNP = hEdge * 0.2
nanopillars = [0.5, 3.0, 3.5]
cell_mesh, facet_markers, cell_markers, substrate_markers, curv_markers, u_nuc, a_nuc = mesh_gen.create_3dcell(
                                                                    contactRad=15.5,
                                                                    hEdge=hEdge, hInnerEdge=hEdge,
                                                                    hNP=hNP,
                                                                    thetaExpr="1",
                                                                    nanopillars=nanopillars,
                                                                    return_curvature=True,
                                                                    sym_fraction=1/2,
                                                                    nuc_compression=args["nuc_compression"],
                                                                    use_tmp=False)
# mesh_folder = args["mesh_folder"]
# mesh_folder.mkdir(exist_ok=True, parents=True)
# mesh_file = mesh_folder / "spreadCell_mesh.h5"
# mesh_tools.write_mesh(cell_mesh, facet_markers, cell_markers, mesh_file, [substrate_markers])
# # save curvatures for reference
# curv_file_name = mesh_folder / "curvatures.xdmf"
# with d.XDMFFile(str(curv_file_name)) as curv_file:
#     curv_file.write(curv_markers)

timer = d.Timer("mechanotransduction-example")
incl_pore = False
no_substrate = False

# # compute nuclear curvature
# NMMesh = d.create_meshview(facet_markers, 12)
# mfcurv = mesh_tools.compute_curvature(ref_mesh = cell_mesh, mf_facet = facet_markers,
#                                       mf_cell = cell_markers, facet_marker_vec = [12],
#                                       cell_marker_vec = [2])
# V = d.FunctionSpace(NMMesh, "P", 1)
# dfunc = d.Function(V)
# bmesh = V.mesh()
# store_map = bmesh.topology().mapping()[cell_mesh.id()].vertex_map()
# values = dfunc.vector().get_local()
# for j in range(len(store_map)):
#     cur_sub_idx = d.vertex_to_dof_map(V)[j]
#     values[cur_sub_idx] = mfcurv.array()[store_map[j]]
# dfunc.vector().set_local(values)
# dfunc.vector().apply("insert")
# -

# Now we define units for use in the model.

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
# stiffness units
kPa = unit.kilopascal

# # Model generation
#
# For each step of model generation, refer to SMART Example 3 or API documentation for further details.
#
# We first define compartments and the compartment container. Note that we can specify nonadjacency for surfaces in the model, which is not required, but can speed up the solution process.

# +
if args["axisymmetric"]:
    Cyto = Compartment("Cyto", 2, um, 1)
    PM = Compartment("PM", 1, um, 10)
    Nuc = Compartment("Nuc", 2, um, 2)
    NM = Compartment("NM", 1, um, 12)
else:
    Cyto = Compartment("Cyto", 3, um, 1)
    PM = Compartment("PM", 2, um, 10)
    Nuc = Compartment("Nuc", 3, um, 2)
    NM = Compartment("NM", 2, um, 12)
PM.specify_nonadjacency(["NM", "Nuc"])
NM.specify_nonadjacency(["PM"])

sa_pm = 1260  # used to compute pFAK and RhoA fluxes
vol_cyto = 2300  # used to compute pFAK and RhoA fluxes
# -

# Compute NPC normalization factor.

loaded = mesh_tools.load_mesh(args["mesh_folder"] / "spreadCell_mesh.h5")
mesh_nm = d.create_meshview(loaded.mf_facet, 12)
zMin = min(mesh_nm.coordinates()[:,2])
zMax = max(mesh_nm.coordinates()[:,2])
centerIdx = np.nonzero(mesh_nm.coordinates()[:,0]==0)[0]
centerIdx = centerIdx[np.argmin(mesh_nm.coordinates()[centerIdx,2])]
zCenter = mesh_nm.coordinates()[centerIdx,2]
if args["npc_slope"] < 0 or args["npc_slope"] > 1:
    raise ValueError("NPC slope must be between 0 and 1")
dx_nm = d.Measure("dx", mesh_nm)
ref_int =  45.74295 #d.assemble_mixed(1.0*dx_nm)
test_expr = d.Expression(f"1 - {args['npc_slope']}*(x[2]-{zMin})/({zMax}-{zMin})", degree=1)
npc_factor = ref_int / d.assemble_mixed(test_expr*dx_nm)
lamin_factor = ref_int / d.assemble_mixed(1.0*dx_nm)

# Compute u_nuc for current geometry.

u_nuc, aInner, bInner = mesh_gen.get_u_nuc(nanopillars[1], max(loaded.mesh.coordinates()[:,2]), 
                                           0.8, args["nuc_compression"], nanopillars)
a_nuc = mesh_gen.compute_stretch(u_nuc, aInner, bInner)
d.ALE.move(u_nuc.function_space().mesh(), u_nuc)

# Now we initialize species. For comparison to ODE results, we can see well_mixed to true (fast diffusion).

# +
z_cutoff = args["z_cutoff"]  # 1e-6 # past this value, no stimulus
if no_substrate:
    substrate_init = "0.0"
elif args["reaction_rate_on_np"] != 1.0:
    substrate_init = (f"1.0*(1-sign(z-{z_cutoff}))/2 + "
                      f"{args['reaction_rate_on_np']}"
                      f"*(1+sign(z-{z_cutoff}))/2")
elif args["curv_sens"]==0:
    substrate_init = "1.0"
else:
    curv0 = args["curv_sens"]
    substrate_init = (f"1.0*(1+({np.sign(curv0)}*sign(curv)))/2 + "
                      f"((1-({np.sign(curv0)}*sign(curv)))/2)*exp(curv/{curv0})")

substrate_A = Species(
    "substrate_A", substrate_init, dimensionless, 0.0, D_unit, "PM"
)  # variable for substrate stiffness stimulation at the boundary
if args["u0_npc"] > 0:
    uNPC = Species("uNPC", 0.0, dimensionless, 0.0, D_unit, "NM")
well_mixed = args["well_mixed"]
init_vals = [0.3, 1.0, 33.6, 0.0, 0.01, 1.5, 0.1, 1.8, 17.9, 482.4, 0.0, 0.0, 0.7, 0.2, 0.7, 0.0]
sp_names = ["pFAK", "RhoA_GDP", "RhoA_GTP", "ROCK_A", "mDia_A", "Myo_A", "LIMK_A", "Cofilin_NP",
            "FActin", "GActin", "LaminA", "NPC_A", "YAPTAZ", "YAPTAZ_phos", "YAPTAZ_nuc"]
if incl_pore:
    sp_names.append("YAPTAZ_nuc_phos")
load_from_file = True
if load_from_file:
    init_vec = []
    new_results_folder = pathlib.Path(str(args['outdir']) + "_append")
    for sp in sp_names:
        cur_file = args['outdir'] / f"{sp}.xdmf"
        if cur_file.is_file():
            init_vec.append(cur_file)
        else:
            init_vec = init_vals
            new_results_folder = args['outdir']
            break
    args['outdir'] = new_results_folder

if well_mixed:
    D_mixed = 100.0
    pFAK = Species("pFAK", init_vec[0], vol_unit, D_mixed, D_unit, "Cyto")
    RhoA_GDP = Species("RhoA_GDP", init_vec[1], vol_unit, D_mixed, D_unit, "Cyto")
    RhoA_GTP = Species("RhoA_GTP", init_vec[2], surf_unit, 0*D_mixed, D_unit, "PM")
    ROCK_A = Species("ROCK_A", init_vec[3], vol_unit, D_mixed, D_unit, "Cyto")
    mDia_A = Species("mDia_A", init_vec[4], vol_unit, D_mixed, D_unit, "Cyto")
    Myo_A = Species("Myo_A", init_vec[5], vol_unit, D_mixed, D_unit, "Cyto")
    LIMK_A = Species("LIMK_A", init_vec[6], vol_unit, D_mixed, D_unit, "Cyto")
    Cofilin_NP = Species("Cofilin_NP", init_vec[7], vol_unit, D_mixed, D_unit, "Cyto")
    FActin = Species("FActin", init_vec[8], vol_unit, D_mixed, D_unit, "Cyto")
    GActin = Species("GActin", init_vec[9], vol_unit, D_mixed, D_unit, "Cyto")
    LaminA = Species("LaminA", init_vec[10], surf_unit, 0*D_mixed, D_unit, "NM")
    NPC_A = Species("NPC_A", init_vec[11], surf_unit, 0*D_mixed, D_unit, "NM")
    YAPTAZ = Species(
        "YAPTAZ", init_vec[12], vol_unit, D_mixed, D_unit, "Cyto"
    )  # non-phosphorylated in the cytosol
    YAPTAZ_phos = Species(
        "YAPTAZ_phos", init_vec[13], vol_unit, D_mixed, D_unit, "Cyto"
    )  # phosphorylated in the cytosol
    YAPTAZ_nuc = Species("YAPTAZ_nuc", init_vec[14], vol_unit, D_mixed, D_unit, "Nuc")
    if incl_pore:
        YAPTAZ_nuc_phos = Species("YAPTAZ_nuc_phos", init_vec[15], vol_unit, D_mixed, D_unit, "Nuc")
else:
    pFAK = Species("pFAK", init_vec[0], vol_unit, 10.0, D_unit, "Cyto")
    RhoA_GDP = Species("RhoA_GDP", init_vec[1], vol_unit, 1.0, D_unit, "Cyto")
    RhoA_GTP = Species("RhoA_GTP", init_vec[2], surf_unit, 0.3, D_unit, "PM")
    ROCK_A = Species("ROCK_A", init_vec[3], vol_unit, 75.0, D_unit, "Cyto")
    mDia_A = Species("mDia_A", init_vec[4], vol_unit, 1.0, D_unit, "Cyto")
    Myo_A = Species("Myo_A", init_vec[5], vol_unit, 0.8, D_unit, "Cyto")
    LIMK_A = Species("LIMK_A", init_vec[6], vol_unit, 10.0, D_unit, "Cyto")
    Cofilin_NP = Species("Cofilin_NP", init_vec[7], vol_unit, 10.0, D_unit, "Cyto")
    FActin = Species("FActin", init_vec[8], vol_unit, 0.6, D_unit, "Cyto")
    GActin = Species("GActin", init_vec[9], vol_unit, 13.37, D_unit, "Cyto")
    LaminA = Species("LaminA", init_vec[10], surf_unit, 0.001, D_unit, "NM")
    NPC_A = Species("NPC_A", init_vec[11], surf_unit, 0.001, D_unit, "NM")
    YAPTAZ = Species(
        "YAPTAZ", init_vec[12], vol_unit, 19.0, D_unit, "Cyto"
    )  # non-phosphorylated in the cytosol
    YAPTAZ_phos = Species(
        "YAPTAZ_phos", init_vec[13], vol_unit, 19.0, D_unit, "Cyto"
    )  # phosphorylated in the cytosol
    YAPTAZ_nuc = Species("YAPTAZ_nuc", init_vec[14], vol_unit, 19.0, D_unit, "Nuc")
    if incl_pore:
        YAPTAZ_nuc_phos = Species("YAPTAZ_nuc_phos", init_vec[15], vol_unit, 19.0, D_unit, "Nuc")
# -

# Now we define reactions and parameters across 3 modules.
#
# Module A: Substrate stiffness -> initial signaling events

# +
# a1: FAK phosphorylation at the membrane
FAK_tot = Parameter("FAK_tot", 1.0, vol_unit)
k_f = Parameter("k_f", 0.015, 1 / sec)
k_sf = Parameter("k_sf", 0.379, 1 / sec)
C = Parameter("C", 3.25, kPa)
cytoConvert = Parameter("cytoConvert", vol_cyto / sa_pm, um)
Emod = Parameter("Emod", args['e_val'], kPa)
a1 = Reaction(
    "a1",
    [],
    ["pFAK"],
    param_map={
        "FAK_tot": "FAK_tot",
        "k_f": "k_f",
        "k_sf": "k_sf",
        "C": "C",
        "cytoConvert": "cytoConvert",
        "Emod": "Emod",
    },
    species_map={"substrate_A": "substrate_A", "pFAK": "pFAK"},
    explicit_restriction_to_domain="PM",
    eqn_f_str=f"cytoConvert*substrate_A*(FAK_tot-pFAK)*(k_f + k_sf*Emod/(C+Emod))")

# a2: FAK dephosphorylation throughout the cytosol
k_df = Parameter("k_df", 0.035, 1 / sec)
a2 = Reaction(
    "a2",
    ["pFAK"],
    [],
    param_map={"k_df": "k_df"},
    species_map={"pFAK": "pFAK"},
    eqn_f_str="k_df*pFAK",
)

# a3: RhoA activation and deactivation
k_fkrho = Parameter("k_fkrho", 0.0168, 1 / sec)
gammaConst = Parameter("gammaConst", 77.56, uM ** (-5))
# n = Parameter("n", 5, dimensionless)
k_drho = Parameter("k_drho", 0.625, 1 / sec)
# RhoA_GDP = Parameter("RhoA_GDP", 1.0, vol_unit)
a3 = Reaction(
    "a3",
    ["RhoA_GDP"],
    ["RhoA_GTP"],
    param_map={
        "k_fkrho": "k_fkrho",
        "gammaConst": "gammaConst",
        "k_drho": "k_drho",
        "cytoConvert": "cytoConvert",
    },
    explicit_restriction_to_domain="PM",
    species_map={"pFAK": "pFAK", "RhoA_GDP": "RhoA_GDP", "RhoA_GTP": "RhoA_GTP"},
    eqn_f_str="RhoA_GDP*cytoConvert*k_fkrho*(gammaConst*pFAK**5 + 1)",
    eqn_r_str="k_drho*RhoA_GTP",
)

# a4: ROCK activation by RhoA_GTP at PM
k_rrho = Parameter("k_rrho", 0.648, 1 / (sec * uM))
ROCK_tot = Parameter("ROCK_tot", 1.0, uM)
a4 = Reaction(
    "a4",
    [],
    ["ROCK_A"],
    param_map={"k_rrho": "k_rrho", "ROCK_tot": "ROCK_tot"},
    species_map={"ROCK_A": "ROCK_A", "RhoA_GTP": "RhoA_GTP"},
    explicit_restriction_to_domain="PM",
    eqn_f_str="k_rrho*RhoA_GTP*(ROCK_tot-ROCK_A)",
)

# a5: Deactivation of ROCK in the cytosol
k_drock = Parameter("k_drock", 0.8, 1 / sec)
a5 = Reaction(
    "a5",
    ["ROCK_A"],
    [],
    param_map={"k_drock": "k_drock"},
    species_map={"ROCK_A": "ROCK_A"},
    eqn_f_str="k_drock*ROCK_A",
)
# -

# Module B: cytoskeletal signaling

# +
# b1: mDia activation by RhoA_GTP at PM
k_mrho = Parameter("k_mrho", 0.002, 1 / (sec * uM))
mDia_tot = Parameter("mDia_tot", 0.8, uM)
b1 = Reaction(
    "b1",
    [],
    ["mDia_A"],
    param_map={"k_mrho": "k_mrho", "mDia_tot": "mDia_tot"},
    species_map={"mDia_A": "mDia_A", "RhoA_GTP": "RhoA_GTP"},
    eqn_f_str="k_mrho*RhoA_GTP*(mDia_tot-mDia_A)",
)

# b2: Deactivation of mDia in the cytosol
k_dmdia = Parameter("k_dmdia", 0.005, 1 / sec)
b2 = Reaction(
    "b2",
    ["mDia_A"],
    [],
    param_map={"k_dmdia": "k_dmdia"},
    species_map={"mDia_A": "mDia_A"},
    eqn_f_str="k_dmdia*mDia_A",
)

# b3: activation and deactivation of myosin
Myo_tot = Parameter("Myo_tot", 5.0, vol_unit)
k_mr = Parameter("k_mr", 0.03, 1 / sec)
ROCK_B = Parameter("ROCK_B", 0.3, vol_unit)
epsilon = Parameter("epsilon", 36.0, 1 / uM)
sc1 = Parameter("sc1", 20.0, 1 / uM)
k_dmy = Parameter("k_dmy", 0.067, 1 / sec)
b3 = Reaction(
    "b3",
    [],
    ["Myo_A"],
    param_map={
        "Myo_tot": "Myo_tot",
        "k_mr": "k_mr",
        "ROCK_B": "ROCK_B",
        "epsilon": "epsilon",
        "sc1": "sc1",
        "k_dmy": "k_dmy",
    },
    species_map={"Myo_A": "Myo_A", "ROCK_A": "ROCK_A"},
    eqn_f_str="(Myo_tot-Myo_A)*k_mr*(1 + epsilon*(ROCK_A/2)*(tanh(sc1*(ROCK_A-ROCK_B)) + 1))\
                         - k_dmy*Myo_A",
)

# b4: activation and deactivation of LIMK
LIMK_tot = Parameter("LIMK_tot", 2.0, vol_unit)
k_lr = Parameter("k_lr", 0.07, 1 / sec)
tau = Parameter("tau", 55.49, 1 / uM)
k_dl = Parameter("k_dl", 2.0, 1 / sec)
b4 = Reaction(
    "b4",
    [],
    ["LIMK_A"],
    param_map={
        "LIMK_tot": "LIMK_tot",
        "k_lr": "k_lr",
        "ROCK_B": "ROCK_B",
        "tau": "tau",
        "sc1": "sc1",
        "k_dl": "k_dl",
    },
    species_map={"LIMK_A": "LIMK_A", "ROCK_A": "ROCK_A"},
    eqn_f_str="(LIMK_tot-LIMK_A)*k_lr*(1 + tau*(ROCK_A/2)*(tanh(sc1*(ROCK_A-ROCK_B)) + 1))\
                         - k_dl*LIMK_A",
)

# b5: dephos. and phos. of Cofilin
Cofilin_tot = Parameter("Cofilin_tot", 2.0, vol_unit)
k_turnover = Parameter("k_turnover", 0.04, 1 / sec)
k_catCof = Parameter("k_catCof", 0.34, 1 / sec)
k_mCof = Parameter("k_mCof", 4.0, uM)
b5 = Reaction(
    "b5",
    [],
    ["Cofilin_NP"],
    param_map={
        "Cofilin_tot": "Cofilin_tot",
        "k_turnover": "k_turnover",
        "k_catCof": "k_catCof",
        "k_mCof": "k_mCof",
    },
    species_map={"LIMK_A": "LIMK_A", "Cofilin_NP": "Cofilin_NP"},
    eqn_f_str="(Cofilin_tot-Cofilin_NP)*k_turnover - k_catCof*LIMK_A*Cofilin_NP/(k_mCof + Cofilin_NP)",
)

# b6: actin polymerization and depolymerization
# Actin_tot = Parameter("Actin_tot", 500.0, uM)
k_ra = Parameter("k_ra", 0.4, 1 / sec)
alpha = Parameter("alpha", 50.0, 1 / uM)
mDia_B = Parameter("mDia_B", 0.165, uM)
k_dep = Parameter("k_dep", 3.5, 1 / sec)
k_fc1 = Parameter("k_fc1", 4.0, 1 / (uM * sec))
b6 = Reaction(
    "b6",
    ["GActin"],
    ["FActin"],
    param_map={
        "k_ra": "k_ra",
        "alpha": "alpha",
        "sc1": "sc1",
        "mDia_B": "mDia_B",
        "k_dep": "k_dep",
        "k_fc1": "k_fc1",
    },
    species_map={
        "FActin": "FActin",
        "GActin": "GActin",
        "mDia_A": "mDia_A",
        "Cofilin_NP": "Cofilin_NP",
    },
    eqn_f_str="GActin*k_ra*(1 + alpha*(mDia_A/2)*(tanh(sc1*(mDia_A-mDia_B)) + 1))\
                         - (k_dep + k_fc1*Cofilin_NP)*FActin",
)
# -

if no_substrate:
    parent_mesh = mesh.ParentMesh(
        mesh_filename=str(pathlib.Path(args["mesh_folder"]) / "spreadCell_mesh.h5"),
        mesh_filetype="hdf5",
        name="parent_mesh",
        # curvature=pathlib.Path(args["mesh_folder"]) / "curvatures.xdmf",
        # extra_keys=["subdomain0_2"]
    )
else:
    parent_mesh = mesh.ParentMesh(
        mesh_filename=str(pathlib.Path(args["mesh_folder"]) / "spreadCell_mesh.h5"),
        mesh_filetype="hdf5",
        name="parent_mesh",
        curvature=pathlib.Path(args["mesh_folder"]) / "curvatures.xdmf",
        extra_keys=["subdomain0_2"]
    )
    substrate_markers = parent_mesh.subdomains[0]
    substrate_A.restrict_to_subdomain(substrate_markers, 10)

# Module C: nucleo-cytoplasmic transport

# +
# Module C: nucleo-cytoplasmic transport
# c1: YAP/TAZ dephos. and phos. in the cytosol
k_CN = Parameter("k_CN", 0.56, 1 / sec)
k_CY = Parameter("k_CY", 0.00076, 1 / (sec * uM**2))
k_NC = Parameter("k_NC", 0.14, 1 / sec)
c1 = Reaction(
    "c1",
    ["YAPTAZ_phos"],
    ["YAPTAZ"],
    param_map={"k_CN": "k_CN", "k_CY": "k_CY", "k_NC": "k_NC"},
    species_map={
        "YAPTAZ": "YAPTAZ",
        "YAPTAZ_phos": "YAPTAZ_phos",
        "FActin": "FActin",
        "Myo_A": "Myo_A",
    },
    eqn_f_str="YAPTAZ_phos*(k_CN + k_CY*FActin*Myo_A) - k_NC*YAPTAZ",
)

# c3: LaminA dephos. and phos.
LaminA_tot = Parameter("LaminA_tot", lamin_factor*3500.0, surf_unit)
k_fl = Parameter("k_fl", 0.46, 1 / sec)
p = Parameter("p", 9.0e-6, kPa / uM**2.6)
C_LaminA = Parameter("C_LaminA", 100.0, kPa)
k_rl = Parameter("k_rl", 0.001, 1 / sec)
c3 = Reaction(
    "c3",
    [],
    ["LaminA"],
    param_map={
        "LaminA_tot": "LaminA_tot",
        "k_fl": "k_fl",
        "p": "p",
        "C_LaminA": "C_LaminA",
        "k_rl": "k_rl",
    },
    explicit_restriction_to_domain="NM",
    species_map={"LaminA": "LaminA", "FActin": "FActin"},
    eqn_f_str="(LaminA_tot - LaminA)*k_fl*p*FActin**2.6/(C_LaminA + p*FActin**2.6) - k_rl*LaminA",
)

# c4: opening and closing of NPCs
NPC_tot = Parameter.from_expression("NPC_tot", f"6.5*{npc_factor}*(1 - {args['npc_slope']}*(z-{zMin})/({zMax-zMin}))", surf_unit)
k_fNPC = Parameter("k_fNPC", 2.8e-7, 1 / (sec * uM**2 * surf_unit))
k_rNPC = Parameter("k_rNPC", 8.7, 1 / sec)
c4 = Reaction(
    "c4",
    [],
    ["NPC_A"],
    param_map={"NPC_tot": "NPC_tot", "k_fNPC": "k_fNPC", "k_rNPC": "k_rNPC"},
    explicit_restriction_to_domain="NM",
    species_map={
        "NPC_A": "NPC_A",
        "LaminA": "LaminA",
        "FActin": "FActin",
        "Myo_A": "Myo_A",
    },
    eqn_f_str="(NPC_tot - NPC_A)*k_fNPC*LaminA*FActin*Myo_A - k_rNPC*NPC_A",
)

# c5: nuclear translocation of YAP/TAZ
k_insolo = Parameter("k_insolo", 1.0, surf_unit / (sec * uM))
k_in2 = Parameter("k_in2", 10.0, 1 / (sec * uM))
k_out = Parameter("k_out", 1.0, surf_unit / (sec * uM))
if args["u0_npc"] > 0:
    u0_NPC = Parameter("u0_NPC", args["u0_npc"], dimensionless)
    c5 = Reaction(
        "c5",
        ["YAPTAZ"],
        ["YAPTAZ_nuc"],
        param_map={"k_insolo": "k_insolo", "k_in2": "k_in2", "k_out": "k_out", "u0_NPC": "u0_NPC"},
        species_map={"YAPTAZ": "YAPTAZ", "YAPTAZ_nuc": "YAPTAZ_nuc", "NPC_A": "NPC_A", "uNPC": "uNPC"},
        explicit_restriction_to_domain="NM",
        eqn_f_str="YAPTAZ*exp((uNPC-1)/u0_NPC)*(k_insolo + k_in2*NPC_A) - k_out*YAPTAZ_nuc",
    )
else:
    c5 = Reaction(
        "c5",
        ["YAPTAZ"],
        ["YAPTAZ_nuc"],
        param_map={"k_insolo": "k_insolo", "k_in2": "k_in2", "k_out": "k_out"},
        species_map={"YAPTAZ": "YAPTAZ", "YAPTAZ_nuc": "YAPTAZ_nuc", "NPC_A": "NPC_A"},
        explicit_restriction_to_domain="NM",
        eqn_f_str="YAPTAZ*(k_insolo + k_in2*NPC_A) - k_out*YAPTAZ_nuc",
    )

# c7: nuclear rupture events
if incl_pore:
    pore = Parameter.from_expression("pore", 
                                     f"(1-exp(-t/1.0))*exp(-(pow(x,2)+pow(y,2)+pow(z-{zCenter},2))/5.0)", 
                                     dimensionless)
    # k_rupture = Parameter("k_rupture", 0.0, 1 / sec)
    # k_rupture = Parameter.from_expression("k_rupture", "(1-exp(-t/1.0))*exp(-pow(x,2)-pow(y,2)-pow(z-3.2,2))", 1 / sec)
    # k_rupture = Parameter.from_expression("k_rupture", "(1-exp(-t/1.0))*exp(-(pow(x,2)+pow(y,2)+pow(z-0.2,2))/5.0)", 1 / sec)
    # k_repair = Parameter("k_repair", 0.1, 1 / sec)
    # pore_max = Parameter("pore_max", 1, surf_unit)
    # c7 = Reaction(
    #     "c7",
    #     [],
    #     ["pore"],
    #     param_map={
    #         "k_rupture": "k_rupture",
    #         "k_repair": "k_repair",
    #         "pore_max": "pore_max",
    #     },
    #     species_map={"pore": "pore"},
    #     explicit_restriction_to_domain="NM",
    #     eqn_f_str="k_rupture*(pore_max-pore) - k_repair*pore",
    # )

    # c7 and c8: YAP leak due to lysis
    k_inPore = Parameter("k_inPore", 100.0, surf_unit / (sec * uM))
    k_outPore = Parameter("k_outPore", 10.0, surf_unit / (sec * uM))
    c7 = Reaction(
        "c7",
        ["YAPTAZ"],
        ["YAPTAZ_nuc"],
        param_map={"k_inPore": "k_inPore", "k_outPore": "k_outPore", "pore": "pore"},
        species_map={"YAPTAZ": "YAPTAZ", "YAPTAZ_nuc": "YAPTAZ_nuc"},
        explicit_restriction_to_domain="NM",
        eqn_f_str="YAPTAZ*pore*k_inPore - k_outPore*pore*YAPTAZ_nuc",
    )
    c8 = Reaction(
        "c8",
        ["YAPTAZ_phos"],
        ["YAPTAZ_nuc_phos"],
        param_map={"k_inPore": "k_inPore", "k_outPore": "k_outPore", "pore": "pore"},
        species_map={"YAPTAZ_phos": "YAPTAZ_phos", "YAPTAZ_nuc_phos": "YAPTAZ_nuc_phos"},
        explicit_restriction_to_domain="NM",
        eqn_f_str="YAPTAZ_phos*pore*k_inPore - k_outPore*pore*YAPTAZ_nuc_phos",
    )
# -

pc, sc, cc, rc = sbmodel_from_locals(locals().values())


# set config for current run
configCur = config.Config()
configCur.flags.update(
    {
        "allow_unused_components": True,
        "axisymmetric_model": args["axisymmetric"],
    }
)
configCur.solver.update(
    {
        "final_t": 10000.0,
        "initial_dt": args["time_step"],
        "time_precision": 6,
        "use_snes": True,
        "attempt_timestep_restart_on_divergence": True,
        "reset_timestep_for_negative_solution": True,
    }
)

model_cur = model.Model(pc, sc, cc, rc, configCur, parent_mesh)
model_cur.initialize(initialize_solver=True)
# Write initial condition(s) to file
results = dict()
result_folder = args["outdir"]
result_folder.mkdir(exist_ok=True, parents=True)

# Initialize values of nuclear deformation

if args["u0_npc"] > 0:
    # Vcur = d.FunctionSpace(parent_mesh.dolfin_mesh, 
    #                        d.VectorElement("P", parent_mesh.dolfin_mesh.ufl_cell(), 
    #                                        degree = 1, dim = 3))
    # u_nuc_load = d.Function(Vcur)
    # u_nuc_file = d.XDMFFile(str(mesh_folder / "u_nuc.xdmf"))
    # u_nuc_file.read_checkpoint(u_nuc_load, "u_nuc", 0)
    
    uvec = uNPC.u["u"].vector()[:]
    NMCoords = uNPC.V.tabulate_dof_coordinates()[:]
    rVals = np.sqrt(np.power(NMCoords[:,0],2) + np.power(NMCoords[:,1],2))
    aRad = np.max(rVals)
    aRadIdx = np.argmax(rVals)
    u_nuc_coords = u_nuc.function_space().tabulate_dof_coordinates()[0:-1:3,:]
    z0 = max(NMCoords[:,2]) - max(u_nuc_coords[:,2])
    for i in range(len(NMCoords)):
        dist_sq = (np.power(NMCoords[i][0]-u_nuc_coords[:,0],2) + 
                   np.power(NMCoords[i][1]-u_nuc_coords[:,1],2) + 
                   np.power(NMCoords[i][2]-z0-u_nuc_coords[:,2],2))
        uvec[uNPC.dof_map[i]] = a_nuc.vector()[np.argmin(dist_sq)]
        # cur_coords = NMCoords[i]
        # cur_coords[2] -= z0
        # try:
        #     uvec[uNPC.dof_map[i]] = a_nuc(cur_coords) #u_nuc(cur_coords)[2]
        # except:
        #     uvec[uNPC.dof_map[i]] = 0.0
    # bRad = np.max(NMCoords[:,2]) - z0
    # substrate_mesh = d.create_meshview(substrate_markers, 10)
    # substrateCoords = substrate_mesh.coordinates()[:]
    # rPM = np.sqrt(np.power(substrateCoords[:,0],2) + np.power(substrateCoords[:,1],2))
    # for i in range(len(rVals)):
    #     substrateDistVals = np.sqrt(np.power(rVals[i]-rPM,2) + np.power(NMCoords[i,2]-substrateCoords[:,2],2))
    #     if np.min(substrateDistVals) < 2.0:
    #         zEllipse =  z0 - bRad*np.sqrt(1 - rVals[i]**2/aRad**2)
    #         uvec[uNPC.dof_map[i]] = NMCoords[i,2] - zEllipse
    for tag in ["u", "n"]:
        uNPC.u[tag].vector().set_local(uvec)
        uNPC.u[tag].vector().apply("insert")

if model_cur.mpi_comm_world.rank == 0:
    import json
    # Dump config to results folder
    (result_folder / "config.json").write_text(
        json.dumps(
            {
                "solver": configCur.solver.__dict__,
                "flags": configCur.flags.__dict__,
                "reaction_database": configCur.reaction_database,
                "mesh_file": str(args["mesh_folder"]),
                "outdir": str(args["outdir"]),
                "time_step": args["time_step"],
                "e_val": args["e_val"],
                "z_cutoff": args["z_cutoff"],
                "axisymmetric": args["axisymmetric"],
                "well_mixed": args["well_mixed"],
            }
        )
    )

for species_name, species in model_cur.sc.items:
    results[species_name] = d.XDMFFile(
        model_cur.mpi_comm_world, str(result_folder / f"{species_name}.xdmf")
    )
    results[species_name].parameters["flush_output"] = True
    results[species_name].write(model_cur.sc[species_name].u["u"], model_cur.t)
# model_cur.to_pickle(result_folder / "model_cur.pkl")


# +
# Set loglevel to warning in order not to pollute notebook output
smart_logger.setLevel(logging.WARNING)

cytoMesh = model_cur.cc["Cyto"].dolfin_mesh
if args["axisymmetric"]:
    xCyto = d.SpatialCoordinate(cytoMesh)[0]
    dxCyto = d.Measure("dx", domain=cytoMesh)
    int_val = d.assemble_mixed(model_cur.sc["FActin"].u["u"] * xCyto * dxCyto)
    vol_cyto = d.assemble_mixed(xCyto * dxCyto)
else:
    dxCyto = d.Measure("dx", domain=cytoMesh)
    int_val = d.assemble_mixed(model_cur.sc["FActin"].u["u"] * dxCyto)
    vol_cyto = d.assemble_mixed(1.0 * dxCyto)
FActin_vec = np.array([int_val/vol_cyto])

nucMesh = model_cur.cc["Nuc"].dolfin_mesh
if args["axisymmetric"]:
    xNuc = d.SpatialCoordinate(nucMesh)[0]
    dxNuc = d.Measure("dx", domain=nucMesh)
    int_val = d.assemble_mixed(model_cur.sc["YAPTAZ_nuc"].u["u"] * xNuc * dxNuc)
    vol_nuc = d.assemble_mixed(xNuc * dxNuc)
else:
    dxNuc = d.Measure("dx", domain=nucMesh)
    int_val = d.assemble_mixed(model_cur.sc["YAPTAZ_nuc"].u["u"] * dxNuc)
    vol_nuc = d.assemble_mixed(1.0 * dxNuc)
YAPTAZ_nuc_vec = np.array([int_val / vol_nuc])#model_cur.sc["YAPTAZ_nuc"].initial_condition])
# -

# Solve
displayed = False
while True:
    logger.info(f"Solve for time step {model_cur.t}")
    model_cur.monolithic_solve()
    model_cur.adjust_dt()

    if args["axisymmetric"]:
        int_val = d.assemble_mixed(model_cur.sc["FActin"].u["u"] * xCyto * dxCyto)
    else:
        int_val = d.assemble_mixed(model_cur.sc["FActin"].u["u"] * dxCyto)
    current_FActin = np.array([int_val / vol_cyto])
    FActin_vec = np.concatenate((FActin_vec, current_FActin))

    if args["axisymmetric"]:
        int_val = d.assemble_mixed(model_cur.sc["YAPTAZ_nuc"].u["u"] * xNuc * dxNuc)
    else:
        int_val = d.assemble_mixed(model_cur.sc["YAPTAZ_nuc"].u["u"] * dxNuc)
    current_YAPTAZ_nuc = np.array([int_val / vol_nuc])
    YAPTAZ_nuc_vec = np.concatenate((YAPTAZ_nuc_vec, current_YAPTAZ_nuc))

    # Save results for post processing
    for species_name, species in model_cur.sc.items:
        results[species_name].write(model_cur.sc[species_name].u["u"], model_cur.t)

    if model_cur.mpi_comm_world.rank == 0:
        np.savetxt(result_folder / "FActin.txt", FActin_vec.astype(np.float32))
        np.savetxt(result_folder / "YAPTAZ_nuc.txt", YAPTAZ_nuc_vec.astype(np.float32))
        np.savetxt(result_folder / "tvec.txt", np.array(model_cur.tvec).astype(np.float32))
    # End if we've passed the final time
    if model_cur.t >= model_cur.final_t:
        break

logger.info("Done with solve loop")
timer.stop()
timings = d.timings(
    d.TimingClear.keep,
    [d.TimingType.wall, d.TimingType.user, d.TimingType.system],
).str(True)

# +
if model_cur.mpi_comm_world.size > 1:
    d.MPI.comm_world.Barrier()

if model_cur.mpi_comm_world.rank == 0:
    print(timings)
    (result_folder / "timings.txt").write_text(timings)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(model_cur.tvec, FActin_vec)
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("FActin (μM)")
    ax[1].plot(model_cur.tvec, YAPTAZ_nuc_vec)
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("YAPTAZ_nuc (μM)")
    fig.savefig(result_folder / "FActin-YAPTAZ_nuc.png")
