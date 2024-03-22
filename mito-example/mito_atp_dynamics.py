# # ATP dynamics in realistic mitochondridal geometries
#
# Here, we implement the model presented in [Garcia et al, Scientific Reports](https://www.biorxiv.org/content/10.1101/2022.08.16.500715v2.full#ref-39), which describes production of ATP in mitochondria.
#
# The geometry in this model is divided into 4 domains - two volumes and two surfaces:
# - outer mitochondrial membrane (OM)
# - intermembrane space (IMS)
# - Inner membrane (IM)
# - Matrix (MAT) (volume inside the mitochondrial matrix)

# +
import dolfin as d
import sympy as sym
import numpy as np
import pathlib
import logging
import gmsh  # must be imported before pyvista if dolfin is imported first
import copy

from smart import config, mesh, model, mesh_tools, visualization
from smart.units import unit
from smart.model_assembly import (
    Compartment,
    Parameter,
    Reaction,
    Species,
    SpeciesContainer,
    ParameterContainer,
    CompartmentContainer,
    ReactionContainer,
    sbmodel_from_locals
)

from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

import sys
import argparse
from mito_parser_args import add_run_mito_arguments
here = pathlib.Path.cwd()
sys.path.insert(0, (here / ".." / "scripts").as_posix())
import runner as main_run

smart_logger = logging.getLogger("smart")
smart_logger.setLevel(logging.DEBUG)
logger = logging.getLogger("mito")
logger.setLevel(logging.INFO)
logger.info("Starting mito example")
# -

# Argument parsing

parser = argparse.ArgumentParser()
add_run_mito_arguments(parser)
single_compartment_im = True
try: # run as python script on cluster
    args = vars(parser.parse_args())
    result_folder = args["outdir"]
    result_folder.mkdir(exist_ok=True)
    # save log to file
    file_handler = logging.FileHandler(f"{str(args['outdir'])}/output.log")
    file_handler.setFormatter(logging.Formatter(config.base_format))
    logger.addHandler(file_handler)
except: # not a run on the cluster
    result_folder = pathlib.Path("results")
    mesh_file = (here / ".." / "meshes" / "mito1_coarser2_mesh.xml")
    new_mesh = pathlib.Path("mesh/mito_mesh.h5")
    curv_file = (here / ".." / "meshes" / "mito1_coarser2_curvature.xml")
    new_curv = pathlib.Path("mesh/mito_curv.xdmf")
    main_run.preprocess_mito_mesh(
        input_mesh_file=mesh_file, output_mesh_file=new_mesh,
        input_curv_file=curv_file, output_curv_file=new_curv,
        dry_run=False, num_refinements=0, single_compartment_im=single_compartment_im)
    args = {}
    args["mesh_file"] = new_mesh
    args["curv_file"] = new_curv
    args["outdir"] = result_folder
    args["time_step"] = 2e-5
    args["curv_dep"] = 0
    args["D"] = 15.0
    result_folder.mkdir(exist_ok=True)

# First, we define the various units for the inputs and define universal constants.

# +
# Aliases - base units
mM = unit.mM
um = unit.um
molecule = unit.molecule
sec = unit.sec
dimensionless = unit.dimensionless
# Aliases - units used in model
D_unit = um**2 / sec
flux_unit = mM * um / sec
vol_unit = mM
surf_unit = molecule / um**2

F = 9.649e4 # Faraday's constant (C/mol)
R = 8.315 # gas constant (J/mol-K)
# -

# ## Model generation
#
# We first define compartments and the add them to a compartment container. Note that we can specify nonadjacency for surfaces in the model, which is not required, but can speed up the solution process.

# +
IMS = Compartment("IMS", 3, um, 1)
OM = Compartment("OM", 2, um, 10)
Mat = Compartment("Mat", 3, um, 2)
Cristae = Compartment("Cristae", 2, um, 11)
IM = Compartment("IM", 2, um, 12)
OM.specify_nonadjacency(["IM", "Mat", "Cristae"])
IM.specify_nonadjacency(["OM"])
Cristae.specify_nonadjacency(["OM"])

cc = CompartmentContainer()
cc.add([IMS, OM, Mat, IM, Cristae])
# -

# Define all model species and and place in species container.

# +
# ATP synthase states
if single_compartment_im:
    ECompartment = "IM"
    EArea = 1.0197 + 0.5256
    LArea = 1.0197 + 0.5256
else:
    ECompartment = "Cristae"
    EArea = 1.0197
    LArea = 0.5256

E_tot = Parameter("E_tot", 267/EArea, surf_unit)
L_tot = Parameter("L_tot", 16471/LArea, surf_unit)
init_from_file = False
init_dict = {"E_Mat": 0.0, "E_IMS": E_tot.value, "E_Mat_H3Star": 0, "E_Mat_H3S": 0, "E_Mat_H3": 0,
                "L": L_tot.value, "TL": 0, "LTsp": 0, "DL": 0, "LD": 0, "TLD": 0, "DLT": 0, "DLD": 0,
                "D_Mat": 0.45*0.8*2.0, "T_Mat": 0.05*13.0, "T_IMS": 0.05*6.5}
if init_from_file:
    init_dir = "/root/scratch/smart-comp-sci-data/mito/results_fixedmodel_coarse/curvlinneg10_coarse2"
    it_init = 2470
    for key, val in init_dict.items():
        init_dict[key] = pathlib.Path(f"{init_dir}/{key}_{it_init}.h5")
    
E_Mat = Species("E_Mat", init_dict["E_Mat"], surf_unit, 0.0, D_unit, ECompartment)
E_IMS = Species("E_IMS", init_dict["E_IMS"], surf_unit, 0.0, D_unit, ECompartment)
E_Mat_H3Star = Species("E_Mat_H3Star", init_dict["E_Mat_H3Star"], surf_unit, 0.0, D_unit, ECompartment)
E_Mat_H3S = Species("E_Mat_H3S", init_dict["E_Mat_H3S"], surf_unit, 0.0, D_unit, ECompartment)
E_Mat_H3 = Species("E_Mat_H3", init_dict["E_Mat_H3"], surf_unit, 0.0, D_unit, ECompartment)
# ANT states
L = Species("L", init_dict["L"], surf_unit, 0.0, D_unit, "IM")
TL = Species("TL", init_dict["TL"], surf_unit, 0.0, D_unit, "IM")
LTsp = Species("LTsp", init_dict["LTsp"], surf_unit, 0.0, D_unit, "IM")
DL = Species("DL", init_dict["DL"], surf_unit, 0.0, D_unit, "IM")
LD = Species("LD", init_dict["LD"], surf_unit, 0.0, D_unit, "IM")
TLD = Species("TLD", init_dict["TLD"], surf_unit, 0.0, D_unit, "IM")
DLT = Species("DLT", init_dict["DLT"], surf_unit, 0.0, D_unit, "IM")
DLD = Species("DLD", init_dict["DLD"], surf_unit, 0.0, D_unit, "IM") # DLD + DLD' in original model
# extra 3 species not needed in this case (commented out following 3 lines)
# DLDPrime = Species("DLDPrime", 0.0, surf_unit, 0.0, D_unit, "IM")
# TLT = Species("TLT", 0.0, surf_unit, 0.0, D_unit, "IM") # TLT + TLT' in original model
# TLTPrime = Species("TLTPrime", 0.0, surf_unit, 0.0, D_unit, "IM")
# ATP/ADP in matrix and IMS
D_Mat = Species("D_Mat", init_dict["D_Mat"], vol_unit, args["D"], D_unit, "Mat")
T_Mat = Species("T_Mat", init_dict["T_Mat"], vol_unit, args["D"], D_unit, "Mat")
T_IMS = Species("T_IMS", init_dict["T_IMS"], vol_unit, args["D"], D_unit, "IMS")
D_IMS = Parameter("D_IMS", 0.45*0.1, vol_unit)

# OM species used for visualization previously
# OMSp = Species("OMSp", 1.0, surf_unit, 0.0, D_unit, "OM")
# kdecay = Parameter("kdecay", 0.0, 1/sec)
# om1 = Reaction("om1", ["OMSp"], [], param_map={"k":"kdecay"}, eqn_f_str="k*OMSp")
# -

# Define mitochondrial conditions (temperature, voltage, pH)

pH_Mat = 7.6 # matrix pH
pH_c = 7.2 # cristae pH
T = 310 # body temperature (K)
dPsi = 180 # membrane voltage (mV)
dPsiB_IMS = -50 # phase boundary potential bulk IMS -> IM (mV) (I think this is reversed)
dPsiB_Mat = 0.0 # phase boundary potential from IM -> Mat (mV)
dPsi_m = dPsi + dPsiB_IMS - dPsiB_Mat
F_RT = F / (1000*R*T) # F/(RT) (1/1000)*C/K (1/mV)

# Now, we define parameters and reactions for each portion of the model. First, we define the ATP synthase dynamics:

# +
# E1: Movement of proton binding site in ATP synthase
k_16 = Parameter("k_16", 100.0*np.exp(3*F_RT*dPsi_m/2), 1/sec) # movement of proton binding site Mat->IMS side
k_61 = Parameter("k_61", 4.98e7*np.exp(-3*F_RT*dPsi_m/2), 1/sec) # movement of proton binding site IMS->Mat side
E1 = Reaction("E1", ["E_Mat"], ["E_IMS"], {"on": "k_16", "off": "k_61"})

# E2: bind/release of 3 protons in IMS
k_65 = Parameter("k_65", 1.58e25*(10**(-pH_c))**3, 1/sec) # binding rate of protons in IMS
k_56 = Parameter("k_56", 2.75e5*np.exp(3*F_RT*dPsiB_IMS), 1/sec) # release of protons in IMS
E2 = Reaction("E2", ["E_IMS"], [],
              param_map={"E_tot":"E_tot", "k_56":"k_56", "k_65":"k_65"},
              species_map={"E_Mat":"E_Mat", "E_IMS":"E_IMS", "E_Mat_H3Star":"E_Mat_H3Star", "E_Mat_H3S":"E_Mat_H3S", "E_Mat_H3":"E_Mat_H3"},
              eqn_f_str="k_65*E_IMS - k_56*(E_tot-E_Mat-E_IMS-E_Mat_H3Star-E_Mat_H3S-E_Mat_H3)")

# E3: movement of 3 protons from IMS to Matrix
k_54 = Parameter("k_54", 100.0, 1/sec)
k_45 = Parameter("k_45", 100.0, 1/sec)
E3 = Reaction("E3", [], ["E_Mat_H3Star"],
              param_map={"E_tot":"E_tot", "k_54":"k_54", "k_45":"k_45"},
              species_map={"E_Mat":"E_Mat", "E_IMS":"E_IMS", "E_Mat_H3Star":"E_Mat_H3Star", "E_Mat_H3S":"E_Mat_H3S", "E_Mat_H3":"E_Mat_H3"},
              eqn_f_str="k_54*(E_tot-E_Mat-E_IMS-E_Mat_H3Star-E_Mat_H3S-E_Mat_H3) - k_45*E_Mat_H3Star")

# E4: movement of 3 protons from IMS to matrix without producing ATP
k_52 = Parameter("k_52", 1e-20, 1/sec)
k_25 = Parameter("k_25", 5.85e-30, 1/sec)
E4 = Reaction("E4", [], ["E_Mat_H3"],
              param_map={"E_tot":"E_tot", "k_52":"k_52", "k_25":"k_25"},
              species_map={"E_Mat":"E_Mat", "E_IMS":"E_IMS", "E_Mat_H3Star":"E_Mat_H3Star", "E_Mat_H3S":"E_Mat_H3S", "E_Mat_H3":"E_Mat_H3"},
              eqn_f_str="k_52*(E_tot-E_Mat-E_IMS-E_Mat_H3Star-E_Mat_H3S-E_Mat_H3) - k_25*E_Mat_H3")

# E5: binding/unbinding of ADP-P to ATP synthase
k_43 = Parameter("k_43", 2e3, 1/(mM*sec))
k_34 = Parameter("k_34", 100.0, 1/sec)
E5 = Reaction("E5", ["E_Mat_H3Star", "D_Mat"], ["E_Mat_H3S"],
              {"on": "k_43", "off": "k_34"})

# E6: ATP production
k_32 = Parameter("k_32", 5e3, 1/sec)
k_23 = Parameter("k_23", 5e3, 1/(mM*sec))
E6 = Reaction("E6", ["E_Mat_H3S"], ["E_Mat_H3", "T_Mat"],
              {"on": "k_32", "off": "k_23"})

# E7: detachment of protons in the matrix
k_21 = Parameter("k_21", 40.0*np.exp(3*F_RT*dPsiB_Mat), 1/sec) # unbinding rate of protons in matrix
k_12 = Parameter("k_12", 6.33e24*(10**(-pH_Mat))**3, 1/sec) # should be 100.0 at pH=7.2 binding rate of protons in matrix
E7 = Reaction("E7", ["E_Mat_H3"], ["E_Mat"], {"on": "k_21", "off": "k_12"})
# -

# Now, we define the reactions associated with adenine nucleotide transporters (ANTs):

# +
# L1: Define all kinetic constants for ANTs
koff_Tm = Parameter("koff_Tm", 4e4, 1/sec) # unbinding of matrix ATP
kon_Tm = Parameter("kon_Tm", 6.4e6/1000, 1/(mM*sec)) # binding of matrix ATP
koff_Ti = Parameter("koff_Ti", 200.0, 1/sec) # unbinding of IMS ATP
kon_Ti = Parameter("kon_Ti", 4e5/1000, 1/(mM*sec)) # binding of IMS ATP
koff_Dm = Parameter("koff_Dm", 4e4, 1/sec) # unbinding of matrix ADP
kon_Dm = Parameter("kon_Dm", 4e6/1000, 1/(mM*sec)) # binding of matrix ADP
koff_Di = Parameter("koff_Di", 100.0, 1/sec) # unbinding of IMS ADP
kon_Di = Parameter("kon_Di", 4e6/1000, 1/(mM*sec)) # binding of IMS ADP
ant_rel = 10 # relative rate of ANTs (speeds up ATP increase with 10x, used in Lupe's paper)
k_p = Parameter("k_p", ant_rel*9.2, 1/sec) # ATP transport Mat->IMS (productive)
k_cp = Parameter("k_cp", ant_rel*0.35, 1/sec) # ATP transport IMS -> Mat (nonproductive)

# L1: Binding of matrix ATP to free ANT (L)
L1 = Reaction("L1", ["L", "T_Mat"], ["LTsp"], {"on": "kon_Tm", "off": "koff_Tm"})

# L2: Binding of matrix ADP to free ANT (L)
L2 = Reaction("L2", ["L", "D_Mat"], ["LD"], {"on": "kon_Dm", "off": "koff_Dm"})

# L3: Binding of IMS ATP to free ANT (L)
L3 = Reaction("L3", ["L", "T_IMS"], ["TL"], {"on": "kon_Ti", "off": "koff_Ti"})

# L4: Binding of IMS ADP to free ANT (L)
L4 = Reaction("L4", ["L"], ["DL"], 
              param_map={"on": "kon_Di", "off": "koff_Di", "D_IMS": "D_IMS"},
              eqn_f_str="on*L*D_IMS - off*DL")

# L5: Binding of matrix ATP to TL
L5 = Reaction("L5", ["TL", "T_Mat"], [],
              param_map={"L_tot":"L_tot", "kon_Tm":"kon_Tm", "koff_Tm":"koff_Tm"},
              species_map={"L":"L", "LTsp":"LTsp", "LD":"LD", "TL":"TL", "DL":"DL", "TLD":"TLD", "DLT":"DLT", "DLD":"DLD", "T_Mat":"T_Mat"},
              eqn_f_str="kon_Tm*TL*T_Mat - koff_Tm*(L_tot-L-LTsp-LD-TL-DL-TLD-DLT-DLD)")

# L6: Binding of matrix ADP to TL
L6 = Reaction("L6", ["TL", "D_Mat"], ["TLD"], {"on": "kon_Dm", "off": "koff_Dm"})

# L7: Binding of IMS ATP to LT
L7 = Reaction("L7", ["LTsp", "T_IMS"], [],
              param_map={"L_tot":"L_tot", "kon_Ti":"kon_Ti", "koff_Tm":"koff_Ti"},
              species_map={"L":"L", "LTsp":"LTsp", "LD":"LD", "TL":"TL", "DL":"DL", "TLD":"TLD", "DLT":"DLT", "DLD":"DLD", "T_IMS":"T_IMS"},
              eqn_f_str="kon_Ti*LTsp*T_IMS - koff_Ti*(L_tot-L-LTsp-LD-TL-DL-TLD-DLT-DLD)")

# L8: Binding of IMS ADP to LT
L8 = Reaction("L8", ["LTsp"], ["DLT"], 
              {"on": "kon_Di", "off": "koff_Di", "D_IMS": "D_IMS"},
              eqn_f_str="on*LTsp*D_IMS - off*DLT")

# L9: Binding of matrix ATP to DL
L9 = Reaction("L9", ["DL", "T_Mat"], ["DLT"], {"on": "kon_Tm", "off": "koff_Tm"})

# L10: Binding of matrix ADP to DL
L10 = Reaction("L10", ["DL", "D_Mat"], ["DLD"], {"on": "kon_Dm", "off": "koff_Dm"})

# L11: Binding of IMS ATP to LD
L11 = Reaction("L11", ["LD", "T_IMS"], ["TLD"], {"on": "kon_Ti", "off": "koff_Ti"})

# L12: Binding of IMS ADP to LD
L12 = Reaction("L12", ["LD"], ["DLD"], 
               {"on": "kon_Di", "off": "koff_Di", "D_IMS": "D_IMS"},
               eqn_f_str="on*LD*D_IMS - off*DLD")

# L13: Exchange of ADP for ATP (can go either way)
L13 = Reaction("L13", ["DLT"], ["TLD"], {"on": "k_p", "off": "k_cp"})
# -

# Finally, consider the export of ATP into the cytosol:

k_vdac = Parameter("k_vdac", 1e6/1000, 1/(mM*sec))
VDAC = Parameter("VDAC", 1e4, surf_unit)
# VDAC = Species("VDAC", 1e4, surf_unit, 0, D_unit, "OM")
T_cyto = Parameter("T_cyto", 0.05*6.5, mM)
V1 = Reaction("V1", ["T_IMS"],[],
              explicit_restriction_to_domain="OM",
              param_map={"k_vdac":"k_vdac", "VDAC":"VDAC", "T_cyto":"T_cyto"},
              species_map={"T_IMS":"T_IMS"},
              # param_map={"k_vdac":"k_vdac", "T_cyto":"T_cyto"},
              # species_map={"T_IMS":"T_IMS", "VDAC":"VDAC"},
              eqn_f_str="k_vdac * VDAC * (T_IMS  - T_cyto)")

# ## Create and load in mesh
#
# Here, we load a realistic mitochondrial geometry.

parent_mesh = mesh.ParentMesh(
    mesh_filename=str(args["mesh_file"]),
    mesh_filetype="hdf5",
    name="parent_mesh",
    curvature=args["curv_file"],
)

# Initialize model and solver.

pc, sc, cc, rc = sbmodel_from_locals(locals().values())
config_cur = config.Config()
config_cur.flags.update({"allow_unused_components": True})
config_cur.solver.update(
    {
        "final_t": 1.0,
        "initial_dt": args["time_step"],
        "time_precision": 6,
        "attempt_timestep_restart_on_divergence": True,
        # "reset_timestep_for_negative_solution": True,
    }
)

# Initialize XDMF files for saving results, save model information to .pkl file, then solve the system until `model_cur.t > model_cur.final_t`. Currently test different cases of curvature dependence, with ATP synthase either preferentially localized to negative curvatures, to positive curvatures, or with no curvature sensitivity.

# +
curv_dep = args["curv_dep"]

model_cur = model.Model(pc, sc, cc, rc, config_cur, parent_mesh)
if curv_dep != 0:
    tot_vals = {"E_IMS": E_tot.value, "L": L_tot.value}
    for sp in ["E_IMS", "L"]:
        # sc[sp].initial_condition = f"{tot_vals[sp]}*exp(curv/{curv_dep})"
        # sc[sp].initial_condition_expression = f"{tot_vals[sp]}*exp(curv/{curv_dep})"
        if curv_dep > 0:
            sc[sp].initial_condition = f"{tot_vals[sp]}*(0.1 + 0.9*(curv/{curv_dep})*(1+sign(curv))/2)"
            sc[sp].initial_condition_expression = f"{tot_vals[sp]}*(0.1 + 0.9*(curv/{curv_dep})*(1+sign(curv))/2)"
        else:
            sc[sp].initial_condition = f"{tot_vals[sp]}*(0.1 + 0.9*(curv/{curv_dep})*(1-sign(curv))/2)"
            sc[sp].initial_condition_expression = f"{tot_vals[sp]}*(0.1 + 0.9*(curv/{curv_dep})*(1-sign(curv))/2)"
model_cur.initialize()

if init_from_file:
    model_cur.idx = it_init

# now compute normalization factor for defining E_IMS initial condition
# such that the total number of E_IMS is conserved regardless of curv_dep
if curv_dep != 0 and not init_from_file:
    geo = mesh_tools.load_mesh(filename=args["mesh_file"], mesh=parent_mesh.dolfin_mesh)
    facet_markers = geo.mf_facet
    if single_compartment_im:
        domain_mesh = d.create_meshview(facet_markers, 12)
        dx = {}
        dx["E_IMS"] = d.Measure("dx", domain=domain_mesh)
        dx["L"] = d.Measure("dx", domain=domain_mesh)
    else:
        Edomain_mesh = d.create_meshview(facet_markers, 11)
        Ldomain_mesh = d.create_meshview(facet_markers, 12)
        dx = {}
        dx["E_IMS"] = d.Measure("dx", domain=Edomain_mesh)
        dx["L"] = d.Measure("dx", domain=Ldomain_mesh)
    spmax = {"E_IMS": 1e4, "L": 3e4}
    for sp in ["E_IMS", "L"]:
        spvec = sc[sp].u["u"].vector()
        spvals_all = spvec.get_local()
        dofmap = sc[sp].dof_map # only select dofs associated with E_IMS
        spvals = spvec.get_local()[dofmap]
        sptot_ref = d.assemble_mixed(tot_vals[sp]*dx[sp])
        recalc = True
        numrecalc = 0

        while recalc:
            if numrecalc > 10:
                raise ValueError(f"Requested gradient is too steep for {sp}")
            # compute normalization factor for conservation of ATP synthase number across conditions
            sptot_cur = d.assemble_mixed(sc[sp].u["u"]*dx[sp])
            scale_factor = sptot_ref/sptot_cur
            spvals = spvals*scale_factor
            # sharp cutoff for molecule density at 1e4 um**-2 (10nm by 10nm)
            dense_logic = spvals > spmax[sp]
            if np.any(dense_logic):
                spvals[np.logical_not(dense_logic)] = spvals[np.logical_not(dense_logic)]*2
                spvals[dense_logic] = spmax[sp]
                numrecalc += 1
            else:
                recalc = False
            spvals_all[dofmap] = spvals
            for tag in ["u", "n"]:
                vec = model_cur.sc[sp].u[tag].vector()
                vec.set_local(spvals_all)
                vec.apply("insert")

# Write initial condition(s) to file
results = dict()

if model_cur.mpi_comm_world.rank == 0:
    import json
    # Dump config to results folder
    (result_folder / "config.json").write_text(
        json.dumps(
            {
                "solver": config_cur.solver.__dict__,
                "flags": config_cur.flags.__dict__,
                "reaction_database": config_cur.reaction_database,
                "mesh_file": str(args["mesh_file"]),
                "curv_file": str(args["curv_file"]),
                "outdir": str(args["outdir"]),
                "time_step": args["time_step"],
            }
        )
    )

separate_files = True

for species_name, species in model_cur.sc.items:
    if separate_files:
        results[species_name] = d.XDMFFile(
            model_cur.mpi_comm_world, str(result_folder / f"{species_name}_{model_cur.idx}.xdmf")
        )
    else:
        results[species_name] = d.XDMFFile(
            model_cur.mpi_comm_world, str(result_folder / f"{species_name}.xdmf")
        )
    results[species_name].parameters["flush_output"] = True
    results[species_name].write(model_cur.sc[species_name].u["u"], model_cur.t)

model_cur.to_pickle(result_folder / "model_cur.pkl")

# Set loglevel to warning in order not to pollute notebook output
smart_logger.setLevel(logging.WARNING)

concVec = np.array([sc["T_IMS"].initial_condition])
IMSMesh = model_cur.cc["IMS"].dolfin_mesh
dx_IMS = d.Measure("dx", domain=IMSMesh)
OMMesh = model_cur.cc["OM"].dolfin_mesh
dx_OM = d.Measure("dx", domain=OMMesh)
volume_IMS = d.assemble_mixed(1.0*dx_IMS)
sa_OM = d.assemble_mixed(1.0*dx_OM)
volume_cyto = 0.306
cyto_convert = 1/(6.0221408e5 * volume_cyto)
# Solve
displayed = False
while True:
    logger.info(f"Solve for time step {model_cur.t}")
    dt = float(model_cur.dt)
    # first estimate for T_cyto
    T_cyto_prev = pc["T_cyto"].value
    T_cyto_flux = d.assemble_mixed(k_vdac.value * VDAC.value * (sc["T_IMS"].u["u"] - T_cyto_prev) * dx_OM)
    pc["T_cyto"].value = T_cyto_prev + T_cyto_flux * dt * cyto_convert
    pc["T_cyto"].dolfin_constant.assign(pc["T_cyto"].value)
    # Solve the system
    model_cur.monolithic_solve()
    # update estimate for T_cyto
    T_cyto_flux = d.assemble_mixed(k_vdac.value * VDAC.value * (sc["T_IMS"].u["u"] - pc["T_cyto"].value) * dx_OM)
    pc["T_cyto"].value = T_cyto_prev + T_cyto_flux * dt * cyto_convert
    pc["T_cyto"].value_vector = np.vstack((pc["T_cyto"].value_vector, [float(model_cur.t), pc["T_cyto"].value]))
    pc["T_cyto"].dolfin_constant.assign(pc["T_cyto"].value)
    model_cur.adjust_dt()
    if model_cur.dt > .02:
        model_cur.set_dt(0.02)
    # Save results for post processing
    for species_name, species in model_cur.sc.items:
        if separate_files:
            results[species_name] = d.XDMFFile(
                model_cur.mpi_comm_world, str(result_folder / f"{species_name}_{model_cur.idx}.xdmf")
            )
        results[species_name].parameters["flush_output"] = True
        results[species_name].write(model_cur.sc[species_name].u["u"], model_cur.t)
    int_val = d.assemble_mixed(model_cur.sc["T_IMS"].u["u"]*dx_IMS)
    curConc = np.array([int_val / volume_IMS])
    concVec = np.concatenate((concVec, curConc))
    if model_cur.mpi_comm_world.rank == 0:
        np.savetxt(result_folder / f"tvec.txt", np.array(model_cur.tvec).astype(np.float32))
        np.savetxt(result_folder / f"T_cyto.txt", np.array(pc["T_cyto"].value_vector).astype(np.float32))
    if model_cur.t >= model_cur.final_t:
        break
# -

# Plot concentration of ATP in the matrix over time.

# +
if model_cur.mpi_comm_world.size > 1:
    d.MPI.comm_world.Barrier()

if model_cur.mpi_comm_world.rank == 0: 
    concVec[0] = sc["T_Mat"].initial_condition
    fig, ax = plt.subplots()
    ax.plot(model_cur.tvec, concVec)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ATP concentration (mM)")
    fig.savefig(result_folder / "mito-example.png")
