Zenodo repository for this code: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13952739.svg)](https://doi.org/10.5281/zenodo.13952739)

Zenodo repository containing results and meshes: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13948827.svg)](https://doi.org/10.5281/zenodo.13948827)

# smart-nanopillars

This code simulates mechanotransduction in cells that have spread on nanopillar substrates.
It is associated with the manuscript entitled, "Nanoscale curvature of the plasma membrane regulates mechanoadaptation through nuclear deformation and rupture."

To run the code in this repository, it is necessary to install [SMART (Spatial Modeling Algorithms for Reaction and Transport)](https://github.com/RangamaniLabUCSD/smart.git), specifically version 2.2.3 or later.
See more info about running the code and reproducing the results in the [scripts](scripts) folder.

## Installation

To run the scripts, we advice usage of docker, and the following base image
`ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30`, which after installation of docker, can be started with

```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared -p 8888:8888 --name smart-comp-sci  ghcr.io/scientificcomputing/fenics-gmsh:2024-05-30
```

This should preferably be started from the root of this git repo, as `-v` shared the current directory on your computer with the docker container.

This will launch a terminal with [FEniCS](https://bitbucket.org/fenics-project/dolfin/src/master/) installed.
To install the compatible version of SMART, call

```bash
python3 -m pip install fenics-smart[lab]==2.2.3 -U
```
Alternatively, you can use the provided docker image from smart directly, i.e
```bash
docker run -ti -v $(pwd):/root/shared -w /root/shared -p 8888:8888 --name smart-comp-sci  ghcr.io/rangamanilabucsd/smart-lab:v2.2.3
```

To run notebooks in your browser, call

```bash
jupyter lab --ip 0.0.0.0 --no-browser --allow-root
```

## Repository contents

This repository is organized into several subfolders, `model-files`, `scripts`, and `utils`. The files contained in each are briefly summarized below.

Notably, prior to running any of the examples, meshes must be generated locally or files must be downloaded from this [this repository](https://doi.org/10.5281/zenodo.13948827). Each individual folder of npy files and `simulation_results_2.8indent`should be place in a folder `analysis_data` and all meshes (in separate folders `nanopillars_baseline`, `nanopillars_indent`, and `nanopillars_movenuc`) should be placed in a folder `meshes`.

### model-files
- `mechanotransduction.ipynb`: Main model file providing the specifications for YAP/TAZ signaling in cells on nanopillar substrates, either with or without nuclear deformation.
- `mechanotransduction_nucOnly.ipynb`: Model file considering only the dynamics of nuclear YAP/TAZ transport through rupture-induced pores and/or NPCs (not modeling any upstream events). This requires an associated previous simulation stored locally, such as `simulation_results_2.8indent` found in the Zenodo repository. Currently the file is configured to sweep over several cases of rupture-induced pores that do not involve any changes in upstream signaling events. This saves significant time from running the full simulations through `mechanotransduction.ipynb`
- `mech_parser_args.py`: Contains names and default values for all input arguments needed to run each script. See this file for definitions of all arguments.
- `pre_process_mesh.py`: Script called to generate meshes.
- `pore_well_mixed.py`: Solves for well-mixed approximation of YAP/TAZ transport shown in Fig 5.
- `mech_figs.ipynb`: Used to generate figures shown throughout the paper.
- `interp_meshes.ipynb`: Notebook used to interpolate from partial mesh (e.g., 1/8th) to full cell mesh and/or to interpolate over different time points from those simulated.
- `mesh_testing.ipynb`: Various mesh-related tests, including calculation of nuclear curvature and comparison to sympy expressions, generation of substrate meshes, and printing mesh statistics.

### scripts
Note that this folder contains its own README to describe the workflow for running simulations in this repository. Most of the infrastructure used here is inherited from the `smart-comp-sci` repository.
- `arguments.py`: script for adding arguments using argparse
- `main.py`: python file used to run scripts (see README in scripts folder)
- `runner.py`: all functions used when calling different scripts and/or generating SLURM file
The folder also considers a series of bash scripts used to generate meshes and run simulations shown in the main paper.
- `nanopillar_mesh_gen.sh`: generate meshes for Figs 1-2 (local)
- `nanopillar_mesh_gen_deform.sh`: generate meshes for Figs 3-4 (local)
- `nanopillar_mesh_gen_movenuc.sh`: generate meshes for Fig S2 (local)
- `run_mechanotransduction_nanopillars.sh`: run simulations for Figs 1-2 (on cluster)
- `nuc_move_testing.sh`: run simulations for Fig S2 (on cluster)
- `npc_stretch_testing.sh`: run simulations for Figs 3-4 (on cluster)
- `pore_testing.sh`: run simulations for Fig 5 (on cluster)

### utils
- `smart_analysis.py`: functions used for postprocessing of XDMF files after running simulation (load in vectors and compute spatial averages)
- `smart_plots.mplstyle`: specifications for matplotlib
- `spread_cell_mesh_generation.py`: Functions used for mesh generation

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg