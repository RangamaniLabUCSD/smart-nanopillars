# Scripts

In this folder we provide an entry point for running the simulations.

usage: main.py [-h] [--submit-tscc]
               {convert-notebooks,mechanotransduction-preprocess,mechanotransduction}

    convert-notebooks   Convert notebooks to python files
    mechanotransduction-preprocess
                        Preprocess mesh for mechanotransduction example
    mechanotransduction
                        Run mechanotransduction example

There are currently 5 ways to execute the scripts. 

1. By running the script without any additional flags, e.g
    ```
    python3 main.py mechanotransduction
    ```
    will run the script directly as a normal script
2. You can submit a job to the [`tscc` cluster] (or adjust SLURM script to run on a similar cluster) by passing the `--submit-tscc` flag, e.g
    ```
    python3 main.py --submit-tscc mechanotransduction
    ```
3. You can navigate to the example folders and run the notebooks directly using `jupyter`


## Setup environment 
All the code in this repository depends on [`smart`](https://rangamanilabucsd.github.io/smart), which in turn depends on the [development version of legacy FEniCs](https://bitbucket.org/fenics-project/dolfin/src/master/). While `smart` is a pure python package and can be easily installed with `pip` (i.e `python3 -m pip install fenics-smart`), the development version of FEniCs can be tricky to install, and we recommend to use [docker](https://www.docker.com) for running the code locally.

### Set up environment using docker

We provide a pre-built docker image containing both the development version of FEniCS and smart which you can pull using 
FIXME: Add correct tag once we have a version we would like to use (with mass conservation and fixes)
```
docker pull ghcr.io/rangamanilabucsd/smart:TAG
```
If you prefer to run the code in jupyter notebooks we also provide a docker image for that
```
docker pull ghcr.io/rangamanilabucsd/smart-lab:TAG
```
You can read more about how to initialize a container and running the code in the [smart documentation](https://rangamanilabucsd.github.io/smart/README.html#installation).


## Running the scripts

All the scripts are available as Jupyter notbooks. If you want to run the examples using the `main.py` script in the following folder, you need to convert the notebooks to python files first. 


### Convert notebooks to python files
In order to run the scripts on the cluster we need to first convert the notebooks to python files. To do this we will use `jupytext` which is part of the requirements. To convert all the notebooks into python files you can do
```
python3 main.py convert-notebooks
```
inside this folder (called `scripts`)

#### Mechanotransduction example
**Pre-process**
```
python3 main.py mechanotransduction-preprocess --mesh-folder meshes-mechanotransduction --shape circle --hEdge 0.6 --hInnerEdge 0.6 --num-refinements 0
```

**Running**
```
python3 main.py mechanotransduction --mesh-folder meshes-mechanotransduction --time-step 0.01 --e-val 70000000.0 --z-cutoff 70000000.0 --outdir results-mechanotransduction
```