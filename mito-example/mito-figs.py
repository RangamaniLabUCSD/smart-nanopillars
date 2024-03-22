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

import sys, os, pathlib
sys.path.append("/root/shared/gitrepos/smart-comp-sci/utils")
sys.path.append("/root/shared/gitrepos/smart-comp-sci/mito-example")
import smart_analysis
from mito_ode import mito_ode_calc
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
params = {'axes.labelsize': 12,
            'axes.titlesize': 6,
            'legend.fontsize': 10,
            'xtick.labelsize':10,
            'ytick.labelsize': 10,
            'figure.figsize': (5,4),
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'legend.loc': "best"}
matplotlib.rcParams.update(params)
cur_dir = "/root/shared/gitrepos/smart-comp-sci/mito-example"

if True:#"npy-files-fixed" not in os.listdir(cur_dir):
    files_dir = "/root/scratch/smart-comp-sci-data/mito"
    npy_dir = pathlib.Path("/root/shared/gitrepos/smart-comp-sci/mito-example/npy-files-new2")
    npy_dir.mkdir(exist_ok=True)
    test_folders = ["curvdepneg20", "curvdepneg10", "curvdep0", "curvdep10", "curvdep20", "D150"]#]
    condition_str = []
    for i in range(len(test_folders)):
        results_folder = f"{files_dir}/{test_folders[i]}"
        # mesh_file = "/root/scratch/smart-comp-sci-data/mito/sweep_figure/meshes/mito_mesh.h5"
        mesh_file = "/root/scratch/smart-comp-sci-data/mito/meshes/mito_mesh.h5"

        tVec, results_all = smart_analysis.analyze_all(
            mesh_file=mesh_file, results_path=results_folder, 
            display=False, ind_files=True)
        results_all.insert(0, tVec) # add time as first element in list
        max_length = len(tVec)
        for j in range(len(results_all)):
            if len(results_all[j]) > max_length:
                max_length = len(results_all[j])
        for j in range(len(results_all)):
            num_zeros = max_length - len(results_all[j])
            for k in range(num_zeros):
                results_all[j].append(0)
        np.save(npy_dir / f"{test_folders[i]}_results.npy", results_all)

# +
matrix_vol = 0.01646550900025656
om_area = 0.6320190057300417
ims_vol = 0.020313914966362288
im0_area = 0.5255854703788347
cristae_area = 1.0196892371671493
cyto_vol = 0.306
geoParam = [matrix_vol, ims_vol, cyto_vol]

var_names_all = ["ADP_matrix", "ATP_matrix", "ATP_cyto", "ATP_IMS", "L", "DL", "LD", 
                 "LT", "TL", "TLD", "DLD", "DLDp", "TLT", "TLTp", "E_IMS", "E_mat", 
                 "H3EStar_mat", "H3ES_mat", "H3E_mat", "ADP_IMS"]
# var_names_saved = ["DL", "DLD", "DLT", "ADP_IMS", "ADP_matrix", "E_IMS", "E_mat", "H3E_mat", "H3ES_mat",
#                    "H3EStar_mat", "L", "LD", "LT", "TL", "TLD", "ATP_IMS", "ATP_matrix", "ATP_cyto"]
var_names_saved = ["DLD", "DLT", "DL", "ADP_matrix", "E_IMS", "E_mat", "H3EStar_mat", "H3ES_mat",
                   "H3E_mat", "LD", "LT", "L", "TLD", "TL", "ATP_IMS", "ATP_matrix", "ATP_cyto"]
# curvdep_vec = [10]#[-60, -30, -15, 0, 15, 30, 60]
npyfiles_old = ["curvdep0_results.npy", "curvlin10_results.npy", "curvdep0_fastD_results.npy", "curvlin10_coarse2_results.npy", "curvlinneg10_coarse2_results.npy"]
Tcytofiles_old = ["T_cyto_curvdep0.txt", "T_cyto_curvlin10.txt", "T_cyto_fastD.txt", "T_cyto_curvlin10_coarse2.txt", "T_cyto_curvlinneg10_coarse2.txt"]
old_dir = "/root/shared/gitrepos/smart-comp-sci-data/numpyfiles/mito"
npyfiles = ["curvdep0_results.npy", "curvdep10_results.npy", "curvdep20_results.npy", "curvdepneg10_results.npy", "curvdepneg20_results.npy", "D150_results.npy"]
Tcytofiles = ["T_cyto_curvdep0.txt", "T_cyto_curvdep10.txt", "T_cyto_curvdep20.txt", "T_cyto_curvdepneg10.txt", "T_cyto_curvdepneg20.txt", "T_cyto_D150.txt"]
plot_names = ["ATP_cyto", "ATP_IMS", "ADP_matrix"]
conversions = [cyto_vol * 6.02e5, ims_vol * 6.02e5, matrix_vol * 6.02e5]
for k in [0]:#range(len(plot_names)):
    plot_name = plot_names[k]
    conversion = conversions[k]
    ode_idx = var_names_all.index(plot_name)
    results_idx = var_names_saved.index(plot_name) + 1 # add one because time is first el
    # first, ode solution
    # t_ode, ode_results = mito_ode_calc([0, 0.5], geoParam, 1.0)
    # plt.plot(t_ode, ode_results[:,ode_idx]/conversion, label=f"ODE", linestyle='dashed')
    # plt.plot(t_ode, ode_results[:,ode_idx]/im0_area, label=f"ODE", linestyle='dashed')
    t_ode, ode_results = mito_ode_calc([0, 1.0], geoParam, 10.0)
    plt.plot(t_ode, ode_results[:,ode_idx]/conversion, label=f"ODE", linestyle='dashed')
    # plt.plot(t_ode, ode_results[:,ode_idx]/im0_area, label=f"ODE", linestyle='dashed')
    for i in range(len(npyfiles)):
        if plot_name == "ATP_cyto":
            file_cur = f"{cur_dir}/T_cyto_files_new2/{Tcytofiles[i]}"
            results_cur = np.loadtxt(file_cur)
            plt.plot(results_cur[:,0], results_cur[:,1],label=f"{npyfiles[i][0:-4]}")
        else:
            # file_cur = f"{cur_dir}/npy-files-new/results_curvdep{curvdep_vec[i]}_results.npy"
            file_cur = f"{cur_dir}/npy-files-new2/{npyfiles[i]}"
            results_cur = np.load(file_cur)
            plt.plot(results_cur[0], results_cur[results_idx],label=f"{npyfiles[i][0:-4]}")
    # for i in range(len(npyfiles_old)):
    #     if plot_name == "ATP_cyto":
    #         file_old = f"{old_dir}/T_cyto_files/{Tcytofiles_old[i]}"
    #         results_old = np.loadtxt(file_old)
    #         plt.plot(results_old[:,0], results_old[:,1],label=f"{npyfiles_old[i][0:-4]}")
    #     else:
    #         # file_cur = f"{cur_dir}/npy-files-new/results_curvdep{curvdep_vec[i]}_results.npy"
    #         file_old = f"{old_dir}/npy-files/{npyfiles_old[i]}"
    #         results_old = np.load(file_old)
    #         plt.plot(results_old[0], results_old[results_idx],label=f"{npyfiles_old[i][0:-4]}")
plt.legend()
# results_f = np.loadtxt("/root/shared/gitrepos/T_cyto.txt")
# plt.plot(results_f[:,0], results_f[:,1],label=f"Tcyto final")
plt.ylabel("Free cytosolic ATP (mM)")
plt.xlabel("Time (s)")
plt.xlim([0, 0.1])
plt.ylim([.3, .34])
# plt.savefig("/root/shared/gitrepos/pyplots/cyto_ATP.pdf", format="pdf")
