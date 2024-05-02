# # Calcium dynamics in a cardiomyocyte calcium release unit
#
# Here, we implement the model presented in [Hake et al 2012, Journal of Physiology](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3477749/), which considers calcium dynamics in a small subregion within a cardiomyocyte. Specifically, the geometry of a calcium release unit (CRU) was considered, which includes the junction between T tubules and the sarcoplasmic reticulum (SR). Calcium entry through T-tubules leads to subsequent release from the SR.
#
# The geometry in this model is divided into 7 domains - 4 volumes and 3 surfaces
# - T-tubules volume (TT) - marker=4
# - T-tubules membrane (TTM) - marker=10
# - SR volume (SR) - marker=2
# - SR membrane (SRM) - marker=12
# - Mitochondrial volume (Mito) - marker=3
# - Mitochondrial membrane (MitoM) - marker=11
# - cytosol (Cyto) - marker=1
#
# This model has six species:
# - $\text{Ca}^{2+}$ in the cytosol
# - $\text{Ca}^{2+}$ in the SR
# - $\text{Ca}^{2+}$ in the T-tubules (currently treated as constant, assuming T-tubules are continuous with extracellular space)
# - ATP (acting as a mobile $\text{Ca}^{2+}$ buffer in the cytosol)
# - Calmodulin (acting as a mobile $\text{Ca}^{2+}$ buffer in the cytosol)
# - Troponin (acting as a stationary $\text{Ca}^{2+}$ buffer in the cytosol)
# - Calsequestrin (calcium buffer in the SR)

# +
import dolfin as d
import sympy as sym
import numpy as np
import os
import pathlib
import logging

from smart import config, mesh, model, mesh_tools, visualization
from smart.units import unit
from smart.model_assembly import (
    Compartment,
    Parameter,
    Reaction,
    Species,
    SpeciesContainer,
    CompartmentContainer,
    sbmodel_from_locals,
)
import argparse
import sys
from cru_args import add_run_cru_arguments
here = pathlib.Path.cwd()
sys.path.insert(0, (here / ".." / "scripts").as_posix())
import runner as main_run

from matplotlib import pyplot as plt

smart_logger = logging.getLogger("smart")
smart_logger.setLevel(logging.DEBUG)
logger = logging.getLogger("cru")
logger.setLevel(logging.INFO)
logger.info("Starting CRU example")
# -

# First, we define the various units for the inputs

# Aliases - base units
uM = unit.uM
mM = unit.mM
um = unit.um
nm = unit.nm
molecule = unit.molecule
sec = unit.sec
dimensionless = unit.dimensionless
# Aliases - units used in model
D_unit = nm**2 / sec
flux_unit = uM * um / sec
vol_unit = uM
surf_unit = molecule / um**2
# other units
mole = unit.mole
kelvin = unit.kelvin
joule = unit.joule
current_unit = unit.picoampere
conductance_unit = unit.picosiemens
volt = unit.volt

parser = argparse.ArgumentParser()
add_run_cru_arguments(parser)
try: # run as python script on cluster
    args = vars(parser.parse_args())
except: # not a run on the cluster
    results_folder = pathlib.Path("results_cru")
    mesh_file = (here / ".." / "meshes" / "CRU_mesh.xml")
    new_mesh = pathlib.Path("mesh/cru_mesh.h5")
    main_run.preprocess_spine_mesh(
        input_mesh_file=mesh_file, output_mesh_file=new_mesh,
        dry_run=False, num_refinements=0)
    args = {}
    args["mesh_file"] = new_mesh  
    args["outdir"] = results_folder
    args["time_step"] = 0.0001
    args["no_serca"] = False

# ## Create and load in mesh
#
# Here, we consider an "ellipsoid-in-an-ellipsoid" geometry. The inner ellipsoid represents the SA and the volume between the SA boundary and the boundary of the outer ellipsoid represents the cytosol.

parent_mesh = mesh.ParentMesh(
    mesh_filename=str(args["mesh_file"]),
    mesh_filetype="hdf5",
    name="parent_mesh",
)

# ## Model generation
#
# For each step of model generation, refer to Example 3 or API documentation for further details.
#
# We first define compartments and the compartment container. Note that we can specify nonadjacency for surfaces in the model, which is not required, but can speed up the solution process.

Cyto = Compartment("Cyto", 3, nm, 1)
SR = Compartment("SR", 3, nm, 2)
Mito = Compartment("Mito", 3, nm, 3)
TT = Compartment("TT", 3, nm, 4)
TTM = Compartment("TTM", 2, nm, 10)
MitoM = Compartment("MitoM", 2, nm, 11)
SRM = Compartment("SRM", 2, nm, 12)
TTM.specify_nonadjacency(["SRM", "SR", "MitoM", "Mito"])
SRM.specify_nonadjacency(["MitoM", "Mito", "TTM", "TT"])
MitoM.specify_nonadjacency(["SRM", "SR", "TTM", "TT"])

# Define parameters and reactions in the cytosol
#

# +
# cytosolic species (and some associated parameters)
Ca = Species("Ca", 0.14, uM, 220.0e6, D_unit, "Cyto")
ATP_tot = Parameter("ATP_tot", 455.0, uM)
ATP = Species("ATP", 455.0 - 0.318, uM, 140.0e6, D_unit, "Cyto")
CMDN_tot = Parameter("CMDN_tot", 24.0, uM)
CMDN = Species("CMDN", CMDN_tot.value-0.471, uM, 25.0e6, D_unit, "Cyto")
TRPN_tot = Parameter("TRPN_tot", 70.0, uM)
# restrict TRPN to away from junction TT
TRPN = Species("TRPN", f"({TRPN_tot.value-13.2})/(1+exp(-((x+125)**2+(z+75)**2)/100**2))", uM, 0.0, D_unit, "Cyto")
# SR species
CaSR = Species("CaSR", 1300.0, uM, 220.0e6, D_unit, "SR")
CSQN_tot = Parameter("CSQN_tot", 6390.0, uM)
CSQN = Species("CSQN", CSQN_tot.value-4280.0, uM, 25.0e6, D_unit, "SR")
# SRM species
RyR = Species("RyR", 1.0, dimensionless, 0.0, D_unit, "SRM")

# TT species
# CaTTsp = Species("CaTTsp", 1800.0, uM, 220.0e6, D_unit, "TT")
CaTT = Parameter("CaTT", 1800.0, uM)

# ttdecay = Parameter("ttdecay", 0.0, 1/sec)
# r0 = Reaction("r0", ["CaTTsp"], [], param_map={"on": "ttdecay"}, eqn_f_str="on*CaTTsp")

# calcium fluxes through SRM
# RyR flux (s1)
NO_RyR = Parameter("NO_RyR", 5.0, dimensionless)
g_RyR = Parameter("g_RyR", 26.2e3, nm/sec) # same as value in original model
cThresh = Parameter("cThresh", 0.092*1300.0, uM)
concUnit = Parameter("concUnit", 1.0, uM)
s1 = Reaction("s1", ["CaSR"], ["Ca"],
              species_map={"RyR": "RyR"},
              param_map={"NO_RyR": "NO_RyR", "g_RyR": "g_RyR", "cThresh":"cThresh", "concUnit":"concUnit"},
              # eqn_f_str="((1+dsign((CaSR - cThresh)/concUnit))/2) * NO_RyR*g_RyR*(CaSR - Ca)",
              eqn_f_str="RyR * NO_RyR*g_RyR*(CaSR - Ca)",
              explicit_restriction_to_domain="SRM")
# SERCA flux (s2)
# here, rather than define separate parameters for each detail of this submodel
# we define variables with units and then parse the magnitude and units into final lumped parameters
pH = 7
T = 273.15 + 22 # Kelvin
protonConc = 10**(-pH+6) * uM
S_SERCA = Parameter("S_SERCA", 1.5, dimensionless)
rho_SERCA = Parameter("rho_SERCA", 75, uM)
SR_VtoA = Parameter("SR_VtoA", 307.0, nm)
MgADP = 10.0 * uM # assume this stays roughly constant
MgATP = 2275 * uM # assume this stays roughly constant
Pi = 1000.0 * uM # assume this remains roughly constant
k2p = 2540 * sec**(-1)
k2m = 67.2 * (uM*sec)**(-1)
k3p = 20.5 * sec**(-1)
k3m = 0.149 * (uM*sec)**(-1)
Kd_Ca = 910 * uM
Kd_CaSR = 2240 * uM
Kd_Hi = 3.54e3 * uM**2
Kd_HSR = 1.05e-2 * uM**2 
Kd_H1 = 1.09e-2 * uM
Kd_H = 7.24e-2 * uM
GFactor = np.exp(11900 / (8.31 * T)) * mM**2 # factor has units to cancel out appropriately
GFactor.ito(uM**2)
MgATP_t = MgATP * (k2m * k3m * Kd_Ca**2 * Kd_HSR * GFactor) / (k2p * k3p * Kd_CaSR**2 * Kd_Hi * Kd_H)
Hi = protonConc**2 / Kd_Hi
HSR = protonConc**2 / Kd_HSR
H1 = protonConc / Kd_H1
H = protonConc / Kd_H
a1p = (k2p*MgATP_t/Kd_Ca**2) / (Hi * (1 + MgATP_t*(1 + H1)))
alpha1_plus = Parameter("alpha1_plus", a1p.magnitude, a1p.units)
a2p = (k3p*HSR) / (HSR*(1 + H) + H)
alpha2_plus = Parameter("alpha2_plus", a2p.magnitude, a2p.units)
a1m = (k2m*MgADP*H/Kd_CaSR**2) / (HSR*(1+H) + H)
alpha1_minus = Parameter("alpha1_minus", a1m.magnitude, a1m.units)
a2m = (k3m*Pi*Hi) / (Hi*(1 + MgATP_t*(1+H1)))
alpha2_minus = Parameter("alpha2_minus", a2m.magnitude, a2m.units)
s2 = Reaction("s2", ["Ca"], ["CaSR"],
              param_map={"S_SERCA":"S_SERCA", "rho_SERCA":"rho_SERCA", "SR_VtoA":"SR_VtoA",
                         "a1p":"alpha1_plus", "a2p":"alpha2_plus", 
                         "a1m":"alpha1_minus", "a2m":"alpha2_minus"},
              eqn_f_str="2*S_SERCA*rho_SERCA*SR_VtoA* (Ca**2 *a1p*a2p - CaSR**2 *a1m*a2m)/(Ca**2 *a1p + a2p + CaSR**2 *a1m + a2m)",
              explicit_restriction_to_domain="SRM")


cm_F = (1e-8 / 96.5e3) * unit.microfarad * mole / (unit.coulomb * um**2)
Voltage = Parameter("Voltage", -82e-3, volt)
RT_F = Parameter("RT_F", 8.31*T / 96.5e3, volt)
# calcium fluxes through TTM
# NCX flux (t1)
Vmax = 3.94 * unit.picoampere/unit.picofarad
Jmax_pint = 1*Vmax* cm_F # s_NCX * Vmax * c_m / F
Jmax_pint.ito(flux_unit)
Jmax = Parameter("Jmax", Jmax_pint.magnitude, Jmax_pint.units)
Nai = Parameter("Nai", 14.2e3, uM)
Nao = Parameter("Nao", 140e3, uM)
VFactor1 = Parameter("VFactor1", np.exp(0.35*Voltage.value/RT_F.value), dimensionless)
VFactor2 = Parameter("VFactor2", np.exp((0.35-1)*Voltage.value/RT_F.value), dimensionless)
ksat = Parameter("ksat", 0.27, dimensionless)
KmCao = 1.4e3 #uM
KmNao = 87.5e3 #uM
KmNai = 12e3 #uM
KmCai = 3.6 #uM
denom1 = Parameter("denom1",
                   KmCao * Nai.value**3 + KmNai**3 * CaTT.value + 
                   KmCai * Nao.value**3 * (1 + (Nai.value/KmNai)**3) + 
                   Nai.value**3 * CaTT.value, uM**4)
denom2 = Parameter("denom2",
                   KmNao**3 + KmNai**3 * CaTT.value/KmCai + Nao.value**3,
                   uM**3)
t1 = Reaction("t1", [], ["Ca"],
              param_map={"Jmax":"Jmax", "Nai":"Nai", "Nao":"Nao", "VFactor1":"VFactor1",
                         "VFactor2":"VFactor2", "ksat":"ksat", "CaTT":"CaTT",
                         "denom1":"denom1", "denom2":"denom2"},
              eqn_f_str="Jmax*(Nai**3*CaTT*VFactor1 - Nao**3*Ca*VFactor2)*(1 + ksat*VFactor2)/(denom1 + denom2*Ca)",
              explicit_restriction_to_domain="TTM") 

# membrane pump flux (t2)
Ip_max = 96e-3 * unit.picoampere / unit.picofarad
Jp_max_pint = 0.5 * Ip_max * cm_F # 0.5 * Ip_max * c_m / F
Jp_max_pint.ito(flux_unit)
Jp_max = Parameter("Jp_max", Jp_max_pint.magnitude, Jp_max_pint.units)
Km_pCa = Parameter("Km_pCa", 0.289, uM)
t2 = Reaction("t2", ["Ca"], [],
              param_map = {"Jp_max":"Jp_max", "Km_pCa":"Km_pCa"},
              eqn_f_str="Jp_max*Km_pCa**2 / (Km_pCa**2 + Ca**2)",
              explicit_restriction_to_domain="TTM")
# leak flux (t3)
GCab = 0.7e-3 * unit.millisiemens / unit.microfarad
# 7.2 factor determined to enforce constant calcium at SS
JCab_pint = 0.5*7.2*GCab*cm_F # 0.5 * s_Cab * GCab * c_m / F
JCab_pint.ito(flux_unit / volt)
JCab = Parameter("JCab", JCab_pint.magnitude, JCab_pint.units) 
t3 = Reaction("t3", ["Ca"], [],
              param_map={"JCab":"JCab", "RT_F":"RT_F", "CaTT":"CaTT", "Voltage":"Voltage"},
              eqn_f_str="JCab*(Voltage - (RT_F/2.0)*ln(CaTT/Ca))",
              explicit_restriction_to_domain="TTM") 

# buffering reactions
# ATP buffering in cytosol (b1)
kon_ATP = Parameter("kon_ATP", 225.0, 1/(sec*uM))
koff_ATP = Parameter("koff_ATP", 45e3, 1/sec)
b1 = Reaction("b1", ["Ca","ATP"], [],
              param_map={"kon_ATP":"kon_ATP", "koff_ATP":"koff_ATP", "ATP_tot":"ATP_tot"},
              eqn_f_str="Ca*ATP*kon_ATP - koff_ATP*(ATP_tot - ATP)")
# CMDN buffering in cytosol (b2)
kon_CMDN = Parameter("kon_CMDN", 34, 1/(sec*uM))
koff_CMDN = Parameter("koff_CMDN", 238, 1/sec)
b2 = Reaction("b2", ["Ca","CMDN"], [],
              param_map={"kon_CMDN":"kon_CMDN", "koff_CMDN":"koff_CMDN", "CMDN_tot":"CMDN_tot"},
              eqn_f_str="Ca*CMDN*kon_CMDN - koff_CMDN*(CMDN_tot - CMDN)")
# TRPN buffering in cytosol (b3)
kon_TRPN = Parameter("kon_TRPN", 32.7, 1/(sec*uM))
koff_TRPN = Parameter("koff_TRPN", 19.6, 1/sec)
b3 = Reaction("b3", ["Ca","TRPN"], [],
              param_map={"kon_TRPN":"kon_TRPN", "koff_TRPN":"koff_TRPN", "TRPN_tot":"TRPN_tot"},
              eqn_f_str="Ca*TRPN*kon_TRPN - koff_TRPN*(TRPN_tot - TRPN)")
# CSQN buffering in SR (b4)
kon_CSQN = Parameter("kon_CSQN", 102, 1/(sec*uM)) 
koff_CSQN = Parameter("koff_CSQN", 65e3, 1/sec)
b4 = Reaction("b4", ["CaSR","CSQN"], [],
              param_map={"kon_CSQN":"kon_CSQN", "koff_CSQN":"koff_CSQN", "CSQN_tot":"CSQN_tot"},
              eqn_f_str="CaSR*CSQN*kon_CSQN - koff_CSQN*(CSQN_tot - CSQN)")
# -
# Create a results folder
result_folder = args["outdir"]
result_folder.mkdir(exist_ok=True)

# Now we add all parameters and reactions to their SMART containers.

pc, sc, cc, rc = sbmodel_from_locals(locals().values())

# Initialize model and solver.

# +
configCur = config.Config()
configCur.flags.update({"allow_unused_components": True})
model_cur = model.Model(pc, sc, cc, rc, configCur, parent_mesh)
configCur.solver.update(
    {
        "final_t": 1.0,
        "initial_dt": args["time_step"],
        "time_precision": 10,
        "use_snes": True,
        # "print_assembly": False,
    }
)

if model_cur.mpi_comm_world.rank == 0:
    import json
    # Dump config to results folder
    (result_folder / "config.json").write_text(
        json.dumps(
            {
                "solver": configCur.solver.__dict__,
                "reaction_database": configCur.reaction_database,
            }
        )
    )


model_cur.initialize(initialize_solver=False)

# adjust RyR localization to junctional SR
RyRVals = sc["RyR"].u["u"].vector().get_local()
RyR_dofmap = sc["RyR"].dof_map
RyRCoords = sc["RyR"].V.tabulate_dof_coordinates()
TTMCoords = cc["TTM"].dolfin_mesh.coordinates()
for i in range(len(RyRCoords)):
    cur_dist_vec = np.sqrt((TTMCoords[:,0]-RyRCoords[i,0])**2+
                           (TTMCoords[:,1]-RyRCoords[i,1])**2+
                           (TTMCoords[:,2]-RyRCoords[i,2])**2)
    if np.min(cur_dist_vec) > 20:
        RyRVals[RyR_dofmap[i]] = 0
for tag in ["u", "n"]:
    vec = sc["RyR"].u[tag].vector()
    vec.set_local(RyRVals)
    vec.apply("insert")

model_cur.initialize_discrete_variational_problem_and_solver()
# -

# Initialize XDMF files for saving results, save model information to .pkl file, then solve the system until `model_cur.t > model_cur.final_t`

# +
# Write initial condition(s) to file
results = dict()
for species_name, species in model_cur.sc.items:
    results[species_name] = d.XDMFFile(
        model_cur.mpi_comm_world, str(result_folder / f"{species_name}.xdmf")
    )
    results[species_name].parameters["flush_output"] = True
    results[species_name].write(model_cur.sc[species_name].u["u"], model_cur.t)
model_cur.to_pickle(result_folder / "model_cur.pkl")

# Set loglevel to warning in order not to pollute notebook output
smart_logger.setLevel(logging.WARNING)

cytoMesh = model_cur.cc["Cyto"].dolfin_mesh
dx = d.Measure("dx", domain=cytoMesh)
volume = d.assemble_mixed(1.0 * dx)
concVec = np.array([d.assemble_mixed(model_cur.sc["Ca"].u["u"] * dx)/volume])
SRMesh = model_cur.cc["SR"].dolfin_mesh
dx_SR = d.Measure("dx", domain=SRMesh)
volume_SR = d.assemble_mixed(1.0 * dx_SR)

tvec = np.array([0.0])
# Solve
displayed = False
while True:
    logger.info(f"Solve for time step {model_cur.t}")
    if d.assemble_mixed(model_cur.sc["CaSR"].u["u"] * dx_SR)/volume_SR < cThresh.value:
        pc["g_RyR"].value = 0.0
        pc["g_RyR"].value_vector = np.vstack((pc["g_RyR"].value_vector, 
                                             [float(model_cur.t), pc["g_RyR"].value]))
        pc["g_RyR"].dolfin_constant.assign(pc["g_RyR"].value)
    # Solve the system
    model_cur.monolithic_solve()
    # adjust time step
    model_cur.adjust_dt()
    # Save results for post processing
    for species_name, species in model_cur.sc.items:
        results[species_name].write(model_cur.sc[species_name].u["u"], model_cur.t)

    curConc = np.array([d.assemble_mixed(model_cur.sc["Ca"].u["u"] * dx) / volume])
    concVec = np.concatenate((concVec, curConc))
    tvec = np.concatenate((tvec, np.array([float(model_cur.t)])))
    
    if model_cur.mpi_comm_world.rank == 0:
        np.savetxt(result_folder / "tvec.txt", np.array(model_cur.tvec).astype(np.float32))
        if model_cur.t > 0.025 and not displayed:  # display first time after .025 s
            visualization.plot(model_cur.sc["Ca"].u["u"])
            displayed = True
    # End if we've passed the final time
    if model_cur.t >= model_cur.final_t:
        break
# -

# Plot average cytosolic calcium over time and save graph.

# +
if model_cur.mpi_comm_world.size > 1:
    d.MPI.comm_world.Barrier()

if model_cur.mpi_comm_world.rank == 0:
    fig, ax = plt.subplots()
    ax.plot(tvec, concVec)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cytosolic calcium (μM)")
    fig.savefig(result_folder / "cru-example.png")

