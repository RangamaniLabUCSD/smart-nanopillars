# Download meshes at https://zenodo.org/records/10480304
python3 main.py dendritic-spine-preprocess --input-mesh-file 1spine_PM10_PSD11_ERM12_cyto1_ER2_coarser.xml --output-mesh-file 1spine_mesh_coarser.h5 --num-refinements 0
python3 main.py dendritic-spine-preprocess --input-mesh-file 1spine_PM10_PSD11_ERM12_cyto1_ER2.xml --output-mesh-file 1spine_mesh.h5 --num-refinements 0
python3 main.py dendritic-spine-preprocess --input-mesh-file 2spine_PM10_PSD11_ERM12_cyto1_ER2.xml --output-mesh-file 2spine_mesh.h5 --num-refinements 0

# # Base case
for timestep in 0.0002 0.0001 0.00005 0.00001
do
    python3 main.py dendritic-spine --mesh-file 1spine_mesh_coarser.h5 --time-step $timestep -o results-dendritic-spine/1spine_coarser_timestep_${timestep}
    python3 main.py dendritic-spine --mesh-file 1spine_mesh.h5 --time-step $timestep -o results-dendritic-spine/1spine_timestep_${timestep}
    python3 main.py dendritic-spine --mesh-file 2spine_mesh.h5 --time-step $timestep -o results-dendritic-spine/2spine_timestep_${timestep}
done
