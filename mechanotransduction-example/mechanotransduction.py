# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
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


from smart import config, mesh, model
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
parser = argparse.ArgumentParser()
add_mechanotransduction_arguments(parser)
try: # run as python script on cluster
    args = vars(parser.parse_args())
except: # not a run on the cluster
    mesh_folder = pathlib.Path("mesh")
    results_folder = pathlib.Path("results")
    main_run.pre_preprocess_mech_mesh(
        mesh_folder=mesh_folder, shape="circle", 
        hEdge=0.3, hInnerEdge=0.3, dry_run = False,
        num_refinements=0, full_3d=False)
    args = {}
    args["mesh_folder"] = mesh_folder
    args["outdir"] = results_folder
    args["time_step"] = 0.01
    args["e_val"] = 7e7
    args["z_cutoff"] = 1e-4
    args["axisymmetric"] = True
    args["well_mixed"] = False

timer = d.Timer("mechanotransduction-example")
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

# Now we initialize species. For comparison to ODE results, we can see well_mixed to true (fast diffusion).

z_cutoff = 100  # 1e-6 # past this value, no stimulus
E_init = f"{args['e_val']}*(1-sign(z-{args['z_cutoff']}))/2"
Emod = Species(
    "Emod", E_init, kPa, 0.0, D_unit, "PM"
)  # variable for substrate stiffness stimulation at the boundary
well_mixed = args["well_mixed"]
if well_mixed:
    D_mixed = 100.0
    pFAK = Species("pFAK", 0.3, vol_unit, D_mixed, D_unit, "Cyto")
    RhoA_GDP = Species("RhoA_GDP", 1.0, vol_unit, D_mixed, D_unit, "Cyto")
    RhoA_GTP = Species("RhoA_GTP", 33.6, surf_unit, 0*D_mixed, D_unit, "PM")
    ROCK_A = Species("ROCK_A", 0.0, vol_unit, D_mixed, D_unit, "Cyto")
    mDia_A = Species("mDia_A", 0.0, vol_unit, D_mixed, D_unit, "Cyto")
    Myo_A = Species("Myo_A", 1.5, vol_unit, D_mixed, D_unit, "Cyto")
    LIMK_A = Species("LIMK_A", 0.1, vol_unit, D_mixed, D_unit, "Cyto")
    Cofilin_NP = Species("Cofilin_NP", 1.8, vol_unit, D_mixed, D_unit, "Cyto")
    FActin = Species("FActin", 17.9, vol_unit, D_mixed, D_unit, "Cyto")
    GActin = Species("GActin", 482.4, vol_unit, D_mixed, D_unit, "Cyto")
    LaminA = Species("LaminA", 0.0, surf_unit, 0*D_mixed, D_unit, "NM")
    NPC_A = Species("NPC_A", 0.0, surf_unit, 0*D_mixed, D_unit, "NM")
    YAPTAZ = Species(
        "YAPTAZ", 0.7, vol_unit, D_mixed, D_unit, "Cyto"
    )  # non-phosphorylated in the cytosol
    YAPTAZ_phos = Species(
        "YAPTAZ_phos", 0.2, vol_unit, D_mixed, D_unit, "Cyto"
    )  # phosphorylated in the cytosol
    YAPTAZ_nuc = Species("YAPTAZ_nuc", 0.7, vol_unit, D_mixed, D_unit, "Nuc")
else:
    pFAK = Species("pFAK", 0.3, vol_unit, 10.0, D_unit, "Cyto")
    RhoA_GDP = Species("RhoA_GDP", 1.0, vol_unit, 1.0, D_unit, "Cyto")
    RhoA_GTP = Species("RhoA_GTP", 33.6, surf_unit, 0.3, D_unit, "PM")
    ROCK_A = Species("ROCK_A", 0.0, vol_unit, 75.0, D_unit, "Cyto")
    mDia_A = Species("mDia_A", 0.0, vol_unit, 1.0, D_unit, "Cyto")
    Myo_A = Species("Myo_A", 1.5, vol_unit, 0.8, D_unit, "Cyto")
    LIMK_A = Species("LIMK_A", 0.1, vol_unit, 10.0, D_unit, "Cyto")
    Cofilin_NP = Species("Cofilin_NP", 1.8, vol_unit, 10.0, D_unit, "Cyto")
    FActin = Species("FActin", 17.9, vol_unit, 0.6, D_unit, "Cyto")
    GActin = Species("GActin", 482.4, vol_unit, 13.37, D_unit, "Cyto")
    LaminA = Species("LaminA", 0.0, surf_unit, 0.001, D_unit, "NM")
    NPC_A = Species("NPC_A", 0.0, surf_unit, 0.001, D_unit, "NM")
    YAPTAZ = Species(
        "YAPTAZ", 0.7, vol_unit, 19.0, D_unit, "Cyto"
    )  # non-phosphorylated in the cytosol
    YAPTAZ_phos = Species(
        "YAPTAZ_phos", 0.2, vol_unit, 19.0, D_unit, "Cyto"
    )  # phosphorylated in the cytosol
    YAPTAZ_nuc = Species("YAPTAZ_nuc", 0.7, vol_unit, 19.0, D_unit, "Nuc")

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
    },
    species_map={"Emod": "Emod", "pFAK": "pFAK"},
    explicit_restriction_to_domain="PM",
    eqn_f_str="cytoConvert*(FAK_tot-pFAK)*(k_f + k_sf*Emod/(C+Emod))")

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
LaminA_tot = Parameter("LaminA_tot", 3500.0, surf_unit)
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
NPC_tot = Parameter("NPC_tot", 6.5, surf_unit)
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
c5 = Reaction(
    "c5",
    ["YAPTAZ"],
    ["YAPTAZ_nuc"],
    param_map={"k_insolo": "k_insolo", "k_in2": "k_in2", "k_out": "k_out"},
    species_map={"YAPTAZ": "YAPTAZ", "YAPTAZ_nuc": "YAPTAZ_nuc", "NPC_A": "NPC_A"},
    explicit_restriction_to_domain="NM",
    eqn_f_str="YAPTAZ*(k_insolo + k_in2*NPC_A) - k_out*YAPTAZ_nuc",
)
# -

pc, sc, cc, rc = sbmodel_from_locals(locals().values())


parent_mesh = mesh.ParentMesh(
    mesh_filename=str(pathlib.Path(args["mesh_folder"]) / "spreadCell_mesh.h5"),
    mesh_filetype="hdf5",
    name="parent_mesh",
)

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
    }
)

model_cur = model.Model(pc, sc, cc, rc, configCur, parent_mesh)
model_cur.initialize(initialize_solver=True)
# Write initial condition(s) to file
results = dict()
result_folder = args["outdir"]
result_folder.mkdir(exist_ok=True, parents=True)

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
            }
        )
    )

for species_name, species in model_cur.sc.items:
    results[species_name] = d.XDMFFile(
        model_cur.mpi_comm_world, str(result_folder / f"{species_name}.xdmf")
    )
    results[species_name].parameters["flush_output"] = True
    results[species_name].write(model_cur.sc[species_name].u["u"], model_cur.t)
model_cur.to_pickle(result_folder / "model_cur.pkl")


# Set loglevel to warning in order not to pollute notebook output
smart_logger.setLevel(logging.WARNING)
FActin_vec = np.array([model_cur.sc["FActin"].initial_condition])
YAPTAZ_nuc_vec = np.array([model_cur.sc["YAPTAZ_nuc"].initial_condition])

# Solve
displayed = False
while True:
    logger.info(f"Solve for time step {model_cur.t}")
    model_cur.monolithic_solve()
    model_cur.adjust_dt()

    cytoMesh = model_cur.cc["Cyto"].dolfin_mesh
    if args["axisymmetric"]:
        x = d.SpatialCoordinate(cytoMesh)[0]
        dx = d.Measure("dx", domain=cytoMesh)
        int_val = d.assemble_mixed(model_cur.sc["FActin"].u["u"] * x * dx)
        volume = d.assemble_mixed(x * dx)
    else:
        dx = d.Measure("dx", domain=cytoMesh)
        int_val = d.assemble_mixed(model_cur.sc["FActin"].u["u"] * dx)
        volume = d.assemble_mixed(1.0 * dx)
    current_FActin = np.array([int_val / volume])
    FActin_vec = np.concatenate((FActin_vec, current_FActin))
    np.savetxt(result_folder / "FActin.txt", FActin_vec.astype(np.float32))

    nucMesh = model_cur.cc["Nuc"].dolfin_mesh
    if args["axisymmetric"]:
        x = d.SpatialCoordinate(nucMesh)[0]
        dx = d.Measure("dx", domain=nucMesh)
        int_val = d.assemble_mixed(model_cur.sc["YAPTAZ_nuc"].u["u"] * x * dx)
        volume = d.assemble_mixed(x * dx)
    else:
        dx = d.Measure("dx", domain=nucMesh)
        int_val = d.assemble_mixed(model_cur.sc["YAPTAZ_nuc"].u["u"] * dx)
        volume = d.assemble_mixed(1.0 * dx)
    current_YAPTAZ_nuc = np.array([int_val / volume])
    YAPTAZ_nuc_vec = np.concatenate((YAPTAZ_nuc_vec, current_YAPTAZ_nuc))

    # Save results for post processing
    for species_name, species in model_cur.sc.items:
        results[species_name].write(model_cur.sc[species_name].u["u"], model_cur.t)

    if model_cur.mpi_comm_world.rank == 0: 
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
