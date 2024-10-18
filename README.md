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

This repository is organized into several subfolders, `model-files`, `scripts`, and `utils`.

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg