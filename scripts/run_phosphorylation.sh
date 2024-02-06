# # Create meshes
# for radius in 1.0 2.0 4.0 6.0 8.0 10.0
# do
#     for refinement in 0 1 2
#     do
#         python3 main.py --dry-run phosphorylation-preprocess --curRadius $radius --num-refinements $refinement --mesh-folder "meshes-phosphorylation/curRadius_${radius}_refined_${refinement}"
#     done
# done

# Diffusion and Refinements
for radius in 1.0 2.0 4.0 6.0 8.0 10.0
do
    for refinement in 0 1 2
    do
        for diffusion in 0.01, 0.1, 1, 10, 100
        do
            python3 main.py --dry-run phosphorylation-preprocess --diffusion $diffusion --curRadius $radius --num-refinements $refinement --mesh-folder "meshes-phosphorylation/curRadius_${radius}_refined_${refinement}"
        done
    done
done

# Time steps
for radius in 1.0 2.0 4.0 6.0 8.0 10.0
do
    for timestep in 0.005 0.0025 .001
    do
        python3 main.py --dry-run phosphorylation-preprocess --time-step $timestep --curRadius $radius --mesh-folder "meshes-phosphorylation/curRadius_${radius}_refined_0"
    done
done