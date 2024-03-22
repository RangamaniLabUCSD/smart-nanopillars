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
import smart_analysis
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
params = {'axes.labelsize': 12,
            'axes.titlesize': 6,
            'legend.fontsize': 10,
            'xtick.labelsize':10,
            'ytick.labelsize': 10,
            'figure.figsize': (6,4),
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'legend.loc': "right"}
matplotlib.rcParams.update(params)

cur_dir = "/root/shared/gitrepos/smart-comp-sci/ca2+-examples"
if True:#"npy-files" not in os.listdir(cur_dir):
    spine_results_folder = "/root/scratch/smart-comp-sci-data/dendritic_spine/dendritic-spine-results"
    spine_mesh = "/root/scratch/smart-comp-sci-data/dendritic_spine/meshes/spine_mesh.h5"
    cru_results_folder = "/root/scratch/smart-comp-sci-data/cru/cru-results"
    cru_results_noserca = "/root/scratch/smart-comp-sci-data/cru/cru-results-noserca"
    cru_mesh = "/root/scratch/smart-comp-sci-data/cru/meshes/cru_mesh.h5"
    results_folders = [spine_results_folder, spine_results_folder, spine_results_folder, spine_results_folder,
                       cru_results_folder, cru_results_folder, cru_results_folder,
                       cru_results_noserca, cru_results_noserca, cru_results_noserca]
    mesh_files = [spine_mesh, spine_mesh, spine_mesh, spine_mesh, 
                  cru_mesh, cru_mesh, cru_mesh, 
                  cru_mesh, cru_mesh, cru_mesh]
    tests = ["spine-all", "spine-head", "spine-neck", "spine-shaft",
              "cru-all", "cru-center", "cru-edge",
              "cru-all-noserca", "cru-center-noserca", "cru-edge-noserca"]
    spine_head = [-1000,-1000,0.2,1000,1000,1000]
    spine_neck = [0, -1000, -0.4, 1000, 1000, 0.2]
    spine_shaft = [-1000,-1000,-1000,1000,1000,-0.4]
    cru_center = [-200,-100,-150,200,150,300]
    cru_edge = [-1000, -1000, -1000, -600, 1000, 1000]
    domains = [[], spine_head, spine_neck, spine_shaft, [], cru_center, cru_edge, [], cru_center, cru_edge]
    for i in [7,8,9]:#range(len(tests)):
        results_folder = results_folders[i]
        npy_dir = pathlib.Path(f"/root/shared/gitrepos/smart-comp-sci/ca2+-examples/npy-files")
        npy_dir.mkdir(exist_ok=True)
        mesh_file = mesh_files[i]

        tVec, results_all = smart_analysis.analyze_all(
            mesh_file=mesh_file, results_path=results_folder, display=False,
            subdomain=domains[i])
        results_all.insert(0, tVec) # add time as first element in list
        max_length = len(tVec)
        for j in range(len(results_all)):
            if len(results_all[j]) > max_length:
                max_length = len(results_all[j])
        for j in range(len(results_all)):
            num_zeros = max_length - len(results_all[j])
            for k in range(num_zeros):
                results_all[j].append(0)
        np.save(npy_dir / f"{tests[i]}-results.npy", results_all)

# +
spine_vars = ["Bf", "Bm", "Ca", "CaSA", "NMDAR"]

old_stored_dir = "/root/shared/gitrepos/smart-comp-sci-data/numpyfiles/calcium/npy-files"
spine_files = [f"{cur_dir}/npy-files/spine-head-results.npy", f"{cur_dir}/npy-files/spine-shaft-results.npy",
               f"{old_stored_dir}/spine-head-results.npy", f"{old_stored_dir}/spine-shaft-results.npy"]

f, (ax1, ax2) = plt.subplots(2, 1, sharex=False)

for i in range(len(spine_files)):
    results_spine = np.load(spine_files[i])
    plot_spine1 = "Ca"

    spine_idx1 = spine_vars.index(plot_spine1) + 1
    ax1.plot(results_spine[0], results_spine[spine_idx1],label="Dendritic spine")
    ax1.set_ylim([0, 5])
    # ax1.legend()
    ax1.set_ylabel("Calcium (μM)")

    plot_spine2 = "CaSA"
    spine_idx2 = spine_vars.index(plot_spine2) + 1
    ax2.plot(results_spine[0], results_spine[spine_idx2],label="Dendritic spine")
    # ax2.legend()
    ax2.set_ylabel("SA calcium (μM)")
    ax2.set_ylim([60, 69])
    ax2.set_xlabel("Time (s)")
# plt.savefig("/root/shared/gitrepos/pyplots/spine_both_plots.pdf", format="pdf")

# +
cur_dir = "/root/shared/gitrepos/smart-comp-sci/ca2+-examples"
# cru_vars = ["ATP", "Ca", "CaSR", "CMDN", "CSQN", "TRPN"]
cru_vars = ["ATP", "Ca", "CaSR", "CMDN", "CSQN", "RyR", "SERCA", "TRPN"]
plot_cru = "Ca"
f, (ax1, ax2) = plt.subplots(2, 1, sharex=False)
plot_cru1 = "Ca"
results_cru_withSERCA = np.load(f"{cur_dir}/npy-files/cru-center-results-fixedSERCA.npy")
results_cru_noSERCA = np.load(f"{cur_dir}/npy-files/cru-center-results-noSERCA.npy")
cru_idx1 = cru_vars.index(plot_cru1) + 1
ax1.plot(results_cru_withSERCA[0], results_cru_withSERCA[cru_idx1],label="CRU - with SERCA")
ax1.plot(results_cru_noSERCA[0], results_cru_noSERCA[cru_idx1],label="CRU - no SERCA")
# ax1.set_ylim([0, 2])
ax1.set_xlim([-.01, 0.2])
# ax1.legend()
ax1.set_ylabel("Calcium (μM)")
ax1.set_xlabel("Time (s)")

plot_cru2 = "CaSR"
cru_idx2 = cru_vars.index(plot_cru2) + 1
ax2.plot(results_cru_withSERCA[0], results_cru_withSERCA[cru_idx2],label="CRU - with SERCA")
ax2.plot(results_cru_noSERCA[0], results_cru_noSERCA[cru_idx2],label="CRU - no SERCA")
# ax2.legend()
ax2.set_ylabel("SR calcium (μM)")
ax2.set_ylim([0, 1400])
ax2.set_xlim([-.01, 0.2])
ax2.set_xlabel("Time (s)")
# plt.savefig("/root/shared/gitrepos/pyplots/cru_edge_plot.pdf", format="pdf")
