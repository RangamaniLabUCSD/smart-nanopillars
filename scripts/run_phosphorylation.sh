# # Create meshes
# for radius in 1 #2 4 6 8 10
# do
#     for refinement in 0 1 2 3 4
#     do
#         python3 main.py phosphorylation-preprocess --curRadius $radius --mesh-folder "meshes-phosphorylation/curRadius_${radius}_refined_${refinement}"
#     done
# done

# Create axisymmetric meshes
# for radius in 1 #2 4 6 8 10
# do
#     for hEdge in 0.4 0.2 0.1 0.05 0.025 0.0125 #5 6 7 8
#     do
#         python3 main.py phosphorylation-preprocess --axisymmetric --curRadius $radius --num-refinements 0 --mesh-folder "meshes-phosphorylation/curRadius_${radius}_axisymmetric_hEdge_${hEdge}" --hEdge $hEdge
#     done
# done

# # Diffusion, Refinements
# for radius in 1 #2 4 6 8 10
# do
#     for refinement in 0 1 2 3 4 5 6 7 8
#     do
#         for diffusion in 0.01 #0.1 1.0 10.0 100.0
#         do
#             python3 main.py --submit-tscc phosphorylation --diffusion $diffusion --curRadius $radius --mesh-folder "meshes-phosphorylation/curRadius_${radius}_refined_${refinement}"
#         done
#     done
# done

#Same with axissymetric
for radius in 1 #2 4 6 8 10
do
    for hEdge in 0.4 0.2 0.1 0.05 0.025 0.0125 #3 4 5 6 7 8
    do
        for diffusion in 0.1 #0.1 1.0 10.0 100.0
        do
            python3 main.py phosphorylation --axisymmetric --diffusion $diffusion --curRadius $radius --mesh-folder "meshes-phosphorylation/curRadius_${radius}_axisymmetric_hEdge_${hEdge}" --outdir "results-phosphorylation/curRadius_${radius}_axisymmetric_hEdge_${hEdge}"
        done
    done
done

# Time step
# for radius in 1 2 4 6 8 10
# do
#     for timestep in 0.005 0.0025 .001
#     do
#         python3 main.py --submit-ex3 phosphorylation --time-step $timestep --curRadius $radius --mesh-folder "meshes-phosphorylation/curRadius_${radius}_refined_0"
#     done
# done