# ex3


## Setup environment
Before submitting any scripts we need to set up the environment on the cluster. First we load the installed modules
```
module use /cm/shared/ex3-modules/202309a/defq/modulefiles
module load python-fenics-dolfin-2019.2.0.dev0
```
Next we create a python virtual environment in the root of the repo. 
```
python3 -m venv venv
```
We activate the virtual environment
```
. venv/bin/activate
```
and install the rest of the dependencies that are not already installed
```
python3 -m pip install -r requirements.txt
```

Whenever you want to run a command you also need to make sure that both the modules are loaded, i.e you have executed the command
```
module use /cm/shared/ex3-modules/202309a/defq/modulefiles
module load python-fenics-dolfin-2019.2.0.dev0
```
and you have activated the environment, i.e you have executed 
```
. venv/bin/activate
```

## Convert notebooks to python files
In order to run the scripts on the cluster we need to first convert the notebooks to python files. To do this we will use `jupytext` which is part of the requirements. To convert all the notebooks into python files you can do
```
python3 main.py convert-notebooks
```
inside this folder (called `ex3_scripts`)

## Examples

### Mechanotransduction example

#### Creating the mesh
Before you can run the mechanotransduction example, you need to create the mesh you want to use. For this you can use the command `preprocess-mech-mesh`, e.g

```
python3 main.py preprocess-mech-mesh --shape circle --hEdge 0.6 --hInnerEdge 0.6 --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6
```
will create a circle mesh with an element size of 0.6 and put it in the folder called `meshes-mechanotransduction/circle_hEdge_0.6`. To see all options you can do

```
python3 main.py preprocess-mech-mesh --help
```

#### Running the example locally
To run the mechanotransduction you should use the command `mechanotransduction`, e.g
```
python3 main.py mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6 --time-step 0.01 --e-val 70000000 --axisymmetric
```
To see all options you can again do 
```
python3 main.py mechanotransduction --help
```

#### Submitting the example to the cluster
To submit the example to run on the cluster you also need to pass the flag `submit-ex3`, e.g
```
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6 --time-step 0.01 --e-val 70000000 --axisymmetric
```