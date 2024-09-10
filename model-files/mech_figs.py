# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import sys, os, pathlib
sys.path.append("/root/shared/gitrepos/smart-comp-sci/utils")
sys.path.append("/root/shared/gitrepos/smart-comp-sci/mechanotransduction-example")
import smart_analysis
from mechanotransduction_ode import mechanotransduction_ode_calc
from matplotlib import pyplot as plt
import matplotlib
params = {'axes.labelsize': 12,
            'axes.titlesize': 12,
            'legend.fontsize': 9,
            'xtick.labelsize':10,
            'ytick.labelsize': 10,
            'figure.figsize': (5.5,4),
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'legend.loc': "best"}
matplotlib.rcParams.update(params)
import numpy as np
cur_dir = "/root/shared/gitrepos/smart-nanopillars"

mesh_dir = "/root/scratch/nanopillar-sims/meshes/nanopillars_new"
if True:#"results-redo" not in os.listdir(cur_dir):
    files_dir = "/root/scratch/nanopillar-sims/results_nanopillars_indentation_newStretchCombined"
    npy_dir = pathlib.Path("/root/shared/gitrepos/smart-nanopillars/analysis_data/npy-files-nanopillars-new")
    npy_dir.mkdir(exist_ok=True)
    test_folders = os.listdir(files_dir)
    condition_str = []
    for i in [0]:#range(1,len(test_folders)):
        # subfolders = os.listdir(f"{files_dir}/{test_folders[i]}")
        # if len(subfolders) != 2: # should be 2 folders, one with mesh and one with results
        #     condition_str.append("")
        #     continue
        # mesh_file = "mesh/spreadCell_mesh.h5"
        # for file in os.listdir(f"{files_dir}/{test_folders[i]}"):
        #     if file.endswith(".h5") and "mesh" in file:
        #         mesh_file = f"{files_dir}/{test_folders[i]}/{file}"
        #         break
        for folder in os.listdir(mesh_dir):
            if folder in test_folders[i]:
                mesh_file = f"{mesh_dir}/{folder}/spreadCell_mesh.h5"
                break
        # if "NPCDensityCorrection" not in test_folders[i]:
        #     continue
        if mesh_file == "":
            Warning("Mesh could not be found, skipping to next case")
            condition_str.append("")
            continue
        results_folder = f"{files_dir}/{test_folders[i]}"
        condition_cur = test_folders[i]
        condition_str.append(condition_cur)
        if mesh_file=="" or results_folder=="":
            ValueError("Folders do not match expected structure for analysis")

        if "circle" in results_folder: #results_folder.endswith("symm=0"):
            axisymm = True
        else:
            axisymm = False
        try:
            tVec, results_all = smart_analysis.analyze_all(
                mesh_file=mesh_file, results_path=results_folder, display=False, axisymm=axisymm)
        except:
            Warning("error in analysis, skipping to next case")
            continue
        results_all.insert(0, tVec) # add time as first element in list
        max_length = len(tVec)
        for j in range(len(results_all)):
            if len(results_all[j]) > max_length:
                max_length = len(results_all[j])
        for j in range(len(results_all)):
            num_zeros = max_length - len(results_all[j])
            for k in range(num_zeros):
                results_all[j].append(0)
        np.save(npy_dir / f"{condition_cur}_results.npy", results_all)

# +
stiffness_vec = [0.1, 5.7, 70000000.0]
radiusArray= [
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.5, 0.25, 0.5, 0.25,
 0.0, 0.0, 0.0, 0.0, 0.0]
pitchArray= [
 5.0, 2.5, 1.0, 5.0, 2.5, 1.0,
 5.0, 2.5, 5.0, 2.5,
 0.0, 0.0, 0.0, 0.0, 0.0]
heightArray=[
 1.0, 1.0, 1.0, 3.0, 3.0, 3.0,
 1.0, 1.0, 3.0, 3.0,
 0.0, 0.0, 0.0, 0.0, 0.0]
cellRadArray=[
 20.25, 18.52, 16.55, 19.93, 18.04, 15.39,
 20.01, 17.45, 18.06, 17.64,
 22.48, 18.08, 15.39, 14.18, 12.33]
EModArray=[
 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 
 10000000, 10000000, 10000000, 10000000,
 10000000, 14, 7, 3, 1]
yapData = [
 2.697152, 2.223328, 2.084372, 2.669816, 2.360008, 2.079816,
 2.55696, 2.264736, 2.456508, 2.365188,
 3.320408, 2.285056, 1.806156, 1.555306, 1.43672
]
curv0Array=[5, 10, 0, 0, 0, 2, 1]
nprateArray=[1, 1, 1, 0.3, 0, 1, 1]
npcSlopeArray=[-1, -0.5, 0, 0.5, 1]
# geoParam = [cyto_vol, nuc_vol, pm_area, nm_area, Ac]
var_names_saved = ["Cofilin_NP", "FActin", "GActin", "LaminA", "LIMK_A", "mDia_A", 
                   "Myo_A", "NPC_A", "pFAK", "RhoA_GDP", "RhoA_GTP", "ROCK_A", "Substrate_A", "YAPTAZ", "YAPTAZ_nuc", "YAPTAZ_phos"]
plot_name = "FActin"
results_idx = var_names_saved.index(plot_name) + 1 # add one because time is first el
plot_names = ["YAPTAZ_phos", "YAPTAZ", "YAPTAZ_nuc"]
results_idx = []
for name in plot_names:
    results_idx.append(var_names_saved.index(name) + 1) # add one because time is first el

fig, ax = plt.subplots()

YAPTAZ_ratios = np.zeros([len(cellRadArray), len(curv0Array)])
errors = np.zeros([len(cellRadArray), len(curv0Array)])

for i in [0, 1, 2, 6, 7, 10, 11, 12, 13, 14]:
    if i >= 10:
        file_cur = f"{cur_dir}/npy-files-nanopillars/nanopillars_h{heightArray[i]}_p{pitchArray[i]}_r{radiusArray[i]}_cellRad{cellRadArray[i]}_results.npy"
        results_cur = np.load(file_cur)
        YAPphos = results_cur[results_idx[0]]
        YAPnp = results_cur[results_idx[1]]
        YAPnuc = results_cur[results_idx[2]]
        YAPratio = YAPnuc / (YAPphos + YAPnp)
        YAPTAZ_ratios[i][0] = YAPratio[-1]
        errors[i][0] = (YAPratio[-1] - yapData[i])**2
        plt.plot(results_cur[0], YAPratio,
                 label=f"h{heightArray[i]}_p{pitchArray[i]}_r{radiusArray[i]}_cellRad{cellRadArray[i]}")
    else:
        for j in range(len(curv0Array)):
            try:
                file_cur = f"{cur_dir}/npy-files-nanopillars/nanopillars_h{heightArray[i]}_p{pitchArray[i]}_r{radiusArray[i]}_cellRad{cellRadArray[i]}_nprate{nprateArray[j]}_curvSens{curv0Array[j]}_NPCDensityCorrection_results.npy"
                results_cur = np.load(file_cur)
            except:
                file_cur = f"{cur_dir}/npy-files-nanopillars/nanopillars_h{heightArray[i]}_p{pitchArray[i]}_r{radiusArray[i]}_cellRad{cellRadArray[i]}_nprate{nprateArray[j]}_curvSens{curv0Array[j]}_results.npy"
                results_cur = np.load(file_cur)
            YAPphos = results_cur[results_idx[0]]
            YAPnp = results_cur[results_idx[1]]
            YAPnuc = results_cur[results_idx[2]]
            YAPratio = YAPnuc / (YAPphos + YAPnp)
            YAPTAZ_ratios[i][j] = YAPratio[-1]
            errors[i][j] = (YAPratio[-1] - yapData[i])**2
            plt.plot(results_cur[0], YAPratio,
                    label=f"h{heightArray[i]}_p{pitchArray[i]}_r{radiusArray[i]}_cellRad{cellRadArray[i]}_nprate{nprateArray[j]}_curvSens{curv0Array[j]}")
            # plt.plot(results_cur[0], yapData[i]*np.ones_like(results_cur[0]), linestyle="dashed",
            #         label=f"h{heightArray[i]}_p{pitchArray[i]}_r{radiusArray[i]}_cellRad{cellRadArray[i]}_nprate{nprateArray[j]}_curvSens{curv0Array[j]}")

# plt.legend()
plt.ylabel("YAP/TAZ N/C")
plt.xlabel('Time (s)')
# plt.xlim([0, 500])
# plt.ylim([0.75, 0.95])
# plt.savefig("/root/shared/gitrepos/pyplots/nanopillars/YAPTAZ.pdf", format="pdf")
# -

curvSens = [0, 1/10, 1/5, 1/2, 1]
errorsSel = [np.sum(errors[:,2]), np.sum(errors[:,1]), np.sum(errors[:,0]),
             np.sum(errors[:,5]), np.sum(errors[:,6])]
plt.plot(curvSens, errorsSel, 'bo-')
plt.ylabel("SSE")
plt.xlabel("Curvature sensitivity")

# +
import matplotlib.pyplot as plt
import numpy as np

conditions = []
# conditions.append("Flat 10 GPa")
for i in [0, 1, 2, 6, 7]:
    conditions.append(f"h{heightArray[i]}p{pitchArray[i]}r{radiusArray[i]}")
# conditions.append("Flat 14 kPa")
# conditions.append("Flat 7 kPa")
# conditions.append("Flat 3 kPa")
# conditions.append("Flat 1 kPa")
YAPTAZ_expt = np.array(yapData)
np_means = {
    'Experiments': YAPTAZ_expt[[0,1,2,6,7]],
    'No curv sens': YAPTAZ_ratios[[0,1,2,6,7],2],
    'Low curv sens': YAPTAZ_ratios[[0,1,2,6,7],1],
    'Mod. curv sens': YAPTAZ_ratios[[0,1,2,6,7],0],
    'High curv sens': YAPTAZ_ratios[[0,1,2,6,7],5],
}

x = np.arange(len(conditions))  # the label locations
width = 0.1  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in np_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    # ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('YAP/TAZ N/C')
ax.set_xticks(x + width, conditions)
ax.legend()#loc='upper right')
plt.ylim([0.0, 3.5])
# plt.savefig("/root/shared/gitrepos/pyplots/nanopillars/bargraph.pdf", format="pdf")

# +
stiffness_vec = [0.1, 5.7, 70000000.0]
radiusArray= [
 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
 0.5, 0.25, 0.5, 0.25,
 0.0, 0.0, 0.0, 0.0, 0.0]
pitchArray= [
 5.0, 2.5, 1.0, 5.0, 2.5, 1.0,
 5.0, 2.5, 5.0, 2.5,
 0.0, 0.0, 0.0, 0.0, 0.0]
heightArray=[
 1.0, 1.0, 1.0, 3.0, 3.0, 3.0,
 1.0, 1.0, 3.0, 3.0,
 0.0, 0.0, 0.0, 0.0, 0.0]
cellRadArray=[
 20.25, 18.52, 16.55, 19.93, 18.04, 15.39,
 20.01, 17.45, 18.06, 17.64,
 22.48, 18.08, 15.39, 14.18, 12.33]
EModArray=[
 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 
 10000000, 10000000, 10000000, 10000000,
 10000000, 14, 7, 3, 1]
npcSlopeArray=[-1.0, -0.5, 0, 0.5, 1.0]
# geoParam = [cyto_vol, nuc_vol, pm_area, nm_area, Ac]
var_names_saved = ["Cofilin_NP", "FActin", "GActin", "LaminA", "LIMK_A", "mDia_A", 
                   "Myo_A", "NPC_A", "pFAK", "RhoA_GDP", "RhoA_GTP", "ROCK_A", "Substrate_A", "YAPTAZ", "YAPTAZ_nuc", "YAPTAZ_phos"]
plot_name = "FActin"
results_idx = var_names_saved.index(plot_name) + 1 # add one because time is first el
plot_names = ["YAPTAZ_phos", "YAPTAZ", "YAPTAZ_nuc"]
results_idx = []
for name in plot_names:
    results_idx.append(var_names_saved.index(name) + 1) # add one because time is first el

fig, ax = plt.subplots()

YAPTAZ_ratios = np.zeros([len(cellRadArray), len(curv0Array)])
errors = np.zeros([len(cellRadArray), len(curv0Array)])

for i in [1]:#[0, 1, 2, 6, 7, 10, 11, 12, 13, 14]:
        for j in range(len(npcSlopeArray)):
            file_cur = f"{cur_dir}/analysis_data/npy-files-nanopillars/nanopillars_h{heightArray[i]}_p{pitchArray[i]}_r{radiusArray[i]}_cellRad{cellRadArray[i]}_NPCSlope{npcSlopeArray[j]}_results.npy"
            results_cur = np.load(file_cur)
            YAPphos = results_cur[results_idx[0]]
            YAPnp = results_cur[results_idx[1]]
            YAPnuc = results_cur[results_idx[2]]
            YAPratio = YAPnuc / (YAPphos + YAPnp)
            YAPTAZ_ratios[i][j] = YAPratio[-1]
            errors[i][j] = (YAPratio[-1] - yapData[i])**2
            plt.plot(results_cur[0], YAPratio,
                    label=f"h{heightArray[i]}_p{pitchArray[i]}_r{radiusArray[i]}_cellRad{cellRadArray[i]}_NPCSlope{npcSlopeArray[j]}")
# plt.legend()
plt.ylabel("YAP/TAZ N/C")
plt.xlabel('Time (s)')
plt.legend()
# plt.xlim([0, 500])
# plt.ylim([2.9, 3.2])
# plt.savefig("/root/shared/gitrepos/pyplots/nanopillars/YAPTAZ.pdf", format="pdf")

# +
stiffness_vec = [0.1, 5.7, 70000000.0]
radius=0.5
pitch=3.5
height=3.0
cellRad=15.5
EMod=10000000
# indentationArray = [0.0, 0.2, 0.4, 0.6, 0.8, 
#                     1.0, 1.2, 1.4, 1.6, 1.8, 
#                     2.0, 2.2, 2.4, 2.6, 2.8]
indentationArray = [0.0, 0.4, 0.8, 
                    1.0, 1.4, 1.8, 
                    2.0, 2.4, 2.8]
u0Array=[0, 2.5, 5.0, 7.5]
var_names_saved = ["Cofilin_NP", "FActin", "GActin", "LaminA", "LIMK_A", "mDia_A", 
                   "Myo_A", "NPC_A", "pFAK", "RhoA_GDP", "RhoA_GTP", "ROCK_A", 
                   "Substrate_A", "YAPTAZ", "YAPTAZ_nuc", "YAPTAZ_phos"]
plot_name = "FActin"
results_idx = var_names_saved.index(plot_name) + 1 # add one because time is first el
plot_names = ["YAPTAZ_phos", "YAPTAZ", "YAPTAZ_nuc"]
results_idx = []
for name in plot_names:
    results_idx.append(var_names_saved.index(name) + 1) # add one because time is first el

fig, ax = plt.subplots()

YAPTAZ_ratios = np.zeros([len(u0Array), len(indentationArray)])

for i in range(len(u0Array)):
    for j in range(len(indentationArray)):
        file_cur = f"{cur_dir}/analysis_data/npy-files-combined/nanopillars_indent{indentationArray[j]}_u0_{u0Array[i]}_combined_results.npy"
        try:
            results_cur = np.load(file_cur)
        except:
                continue
        if u0Array[i] > 0: # then additional u0 variable included 
            YAPphos = results_cur[results_idx[0]+1]
            YAPnp = results_cur[results_idx[1]+1]
            YAPnuc = results_cur[results_idx[2]+1]
        else:
            YAPphos = results_cur[results_idx[0]]
            YAPnp = results_cur[results_idx[1]]
            YAPnuc = results_cur[results_idx[2]]
        YAPratio = YAPnuc / (YAPphos + YAPnp)
        YAPTAZ_ratios[i][j] = YAPratio[-1]
        if u0Array[i] == 2.5 and indentationArray[j] == 0.8:
            YAPTAZ_ratio_ref = YAPratio[-1]
            YAPTAZ_dyn_ref = [results_cur[0], YAPratio]
            NPC_ref = results_cur[8][-1]
            phiRef = YAPphos[-1]/(YAPphos[-1] + YAPnp[-1])
        if u0Array[i] == 0:
            plt.plot(results_cur[0], YAPratio,
                label=f"indent{indentationArray[j]}_u0_{u0Array[i]}", linestyle="dashed")
        else:
            plt.plot(results_cur[0], YAPratio,
            label=f"indent{indentationArray[j]}_u0_{u0Array[i]}")
# plt.legend()
plt.ylabel("YAP/TAZ N/C")
plt.xlabel('Time (s)')
plt.legend()
# plt.xlim([0, 200])
# plt.ylim([2.9, 3.2])
# plt.savefig("YAPTAZ_withcompression.pdf", format="pdf")
# -

selIdx1 = np.array([0,1,3,4,5,6,7,8])
plt.plot(np.array(indentationArray)[selIdx1], YAPTAZ_ratios[0][selIdx1], marker="o",label="no stretch sensitivity")
plt.plot(indentationArray, YAPTAZ_ratios[1], marker="o", label="with highly stretch sensitive NPCs")
plt.plot(indentationArray, YAPTAZ_ratios[2], marker="o", label="with moderately stretch sensitive NPCs")
plt.plot(indentationArray, YAPTAZ_ratios[3], marker="o", label="with low stretch sensitive NPCs")
# plt.ylim([2, 3])
# plt.xlim([-.05, 1.55])
plt.xlabel("Indentation (μm)")
plt.ylabel("YAP/TAZ N/C")
plt.legend()
# plt.savefig("indentation_NCsummary.pdf", format="pdf")

# +
stiffness_vec = [0.1, 5.7, 70000000.0]
radius=0.5
pitch=3.5
height=3.0
cellRad=15.5
EMod=10000000

indentation = 0.8#[0.8, 1.8, 2.8]
poreLoc= 0#[0, 0, 0, 3.5, 3.5, 3.5]
poreSizeArray=[0.1, 0.2, 0.5]
poreRateArray=[1, 10, 100]
transportRateArray=[10, 10, 10, 100, 100, 100]
transportRatioArray=[1, 3, 10, 1, 3, 10]

var_names_saved = ["uNPC", "YAPTAZ", "YAPTAZ_nuc", "YAPTAZ_nuc_phos", "YAPTAZ_phos"]
plot_names = ["YAPTAZ_phos", "YAPTAZ", "YAPTAZ_nuc", "YAPTAZ_nuc_phos"]
results_idx = []
for name in plot_names:
    results_idx.append(var_names_saved.index(name) + 1) # add one because time is first el
fig, ax = plt.subplots()

pore_dir = f"{cur_dir}/analysis_data/npy-files-nanopillars-pores"

YAPTAZ_ratios = np.zeros([len(poreSizeArray),len(poreRateArray),len(transportRateArray)])

for i in [2]:#range(len(poreSizeArray)):
    for j in [1]:#range(len(poreRateArray)):
        for k in range(len(transportRateArray)):
            cur_file = (f"nanopillars_indent{indentation}_pore_size{poreSizeArray[i]}_"
                       f"loc{poreLoc}_rate{poreRateArray[j]}_transport{transportRateArray[k]}_"
                       f"ratio{transportRatioArray[k]}_results.npy")
            cur_file = f"{pore_dir}/{cur_file}"
            try:
                results_cur = np.load(cur_file)
            except:
                continue
            YAPphos = results_cur[results_idx[0]]
            YAPnp = results_cur[results_idx[1]]
            YAPnuc = results_cur[results_idx[2]]
            YAPphos_nuc = results_cur[results_idx[3]]
            YAPratio = (YAPnuc + YAPphos_nuc) / (YAPphos + YAPnp)
            plt.plot(results_cur[0], YAPratio, label=f"Entry/exit ratio = {transportRatioArray[k]}")
            YAPTAZ_ratios[i][j][k] = YAPratio[-1]
plt.plot(YAPTAZ_dyn_ref[0], YAPTAZ_dyn_ref[1], linestyle="dashed", label="no pore")
plt.legend()
plt.ylabel("YAP/TAZ N/C")
plt.xlabel('Time (s)')
plt.title("Slow entry through pore")
plt.xlim([0, 3600])
plt.ylim([0, 7.5])
# plt.savefig("YAPTAZ_withcompression.pdf", format="pdf")
# -

selIdx1 = np.array([0,1,3,4,5,6,7,8])
plt.plot(np.array(transportRatioArray[0:3]), YAPTAZ_ratios[2][1][0:3], marker="o",label="slow entry through pore")
plt.plot(np.array(transportRatioArray[3:]), YAPTAZ_ratios[2][1][3:], marker="o",label="fast entry through pore")
plt.plot(np.array(transportRatioArray[0:3]), YAPTAZ_ratio_ref*np.ones_like(YAPTAZ_ratios[2][1][0:3]), 
         linestyle="dashed",label="no pore")
# theoretical predictions for well mixed
ratios = np.linspace(1, 10, 100)
kin = 1.0 + 10.0*NPC_ref
kout = 1
kout_pore = 10
kin_pore = kout_pore*ratios
SAnuc = 390
SApore = 2*np.pi*.5**2
phiPhos = phiRef
# pred1 = ratios*(kin*(1+zeta1-phiRef) + kout*ratios*phiRef)/(zeta1*kin + kout*ratios)
pred1 = ratios*phiPhos*(1+np.tanh(SApore*kout_pore/100))/2 + (SAnuc*kin*(1-phiPhos) + SApore*kin_pore*(1-phiPhos))/(SAnuc*kout + SApore*kout_pore)
plt.plot(ratios, pred1)
# plt.plot(np.array(transportRatioArray[0:3]), YAPTAZ_ratios[0][1][0:3], marker="o",label="transport = 10")
# plt.plot(np.array(transportRatioArray[3:]), YAPTAZ_ratios[0][1][3:], marker="o",label="transport = 100")
# plt.ylim([2, 3])
# plt.xlim([-.05, 1.55])
plt.xlabel("Transport ratio")
plt.ylabel("YAP/TAZ N/C")
plt.legend()

import dolfin as d
from smart import mesh_tools
import pathlib
import numpy as np
loaded = mesh_tools.load_mesh(pathlib.Path("mesh") / "spreadCell_mesh.h5")
mesh_cur = d.create_meshview(loaded.mf_facet, 12)
Vcur = d.FunctionSpace(mesh_cur, "P", 1)
centerIdx = np.nonzero(mesh_cur.coordinates()[:,0]==0)[0]
centerIdx = centerIdx[np.argmin(mesh_cur.coordinates()[centerIdx,2])]
zCenter = mesh_cur.coordinates()[centerIdx,2]
tVals = np.linspace(0., 5., 200)
pore_xdmf = d.XDMFFile(d.MPI.comm_world, "pore.xdmf")
pore_xdmf.parameters["flush_output"] = True
t = d.Constant(tVals[0])
pore_expr = d.Expression(f"(1-exp(-tCur/1.0))*exp(-(pow(x[0],2)+pow(x[1],2)+pow(x[2]-{zCenter},2))/5.0)", tCur = t, degree=1)
pore_fcn = d.interpolate(pore_expr, Vcur)
for i in range(len(tVals)):
    t.assign(tVals[i])
    pore_fcn = d.interpolate(pore_expr, Vcur)
    pore_fcn.rename("pore", "pore")
    pore_xdmf.write(pore_fcn, tVals[i])

from matplotlib import pyplot as plt
tVals = np.linspace(0., 5., 200)
poreVals = (1-np.exp(-tVals/1.0))
plt.xlabel("Time (s)")
plt.ylabel("permeability")
plt.plot(tVals, poreVals)
