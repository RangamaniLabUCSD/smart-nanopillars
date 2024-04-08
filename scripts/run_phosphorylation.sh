# # Create meshes
# for radius in 1 2 4 6 8 10
# do
#     for refinement in 0 1 2
#     do
#         python3 main.py phosphorylation-preprocess --curRadius $radius --mesh-folder "meshes-phosphorylation/curRadius_${radius}_refined_${refinement}"
#     done
# done

# # Create axisymmetric meshes
# for radius in 1 2 4 6 8 10
# do
#     for refinement in 0 1 2
#     do
#         python3 main.py phosphorylation-preprocess --axisymmetric --curRadius $radius --num-refinements $refinement --mesh-folder "meshes-phosphorylation/curRadius_${radius}_axisymmetric_refined_${refinement}"
#     done
# done

partition=defq

# # Diffusion, Refinements
for radius in 1 2
do
    for refinement in 1 2 3 4
    do
        for diffusion in 0.01 10.0
        do
            for time_step in 0.01 0.1 0.5 1.0
            do
                python3 main.py --submit-ex3 --partition=$partition  phosphorylation --diffusion $diffusion --curRadius $radius --time-step $time_step --mesh-folder  "meshes-phosphorylation/curRadius_${radius}_refined_${refinement}"
                sleep 5
            done
        done
    done
done

# # Run with MPI for D = 10.0, Raidus = 1 and 2 levels of refinement
for refinement in 0 3
do
    for radius in 1
    do
        for ntasks in 2 4 6 8 10 12 14 16
        do
            python3 main.py --submit-ex3 --ntasks $ntasks --partition=$partition  phosphorylation --diffusion 10.0 --curRadius $radius --mesh-folder "meshes-phosphorylation/curRadius_${radius}_refined_${refinement}"
            sleep 5
        done
    done
done