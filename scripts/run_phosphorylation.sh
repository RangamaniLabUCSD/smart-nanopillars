# Create meshes
for radius in 1 2 4 6 8 10
do
    for refinement in 0 1 2
    do
        python3 main.py phosphorylation-preprocess --curRadius $radius --mesh-folder "meshes-phosphorylation/curRadius_${radius}_refined_${refinement}"
    done
done

# Create axisymmetric meshes
for radius in 1 2 4 6 8 10
do
    for refinement in 0 1 2
    do
        python3 main.py phosphorylation-preprocess --axisymmetric --curRadius $radius --num-refinements $refinement --mesh-folder "meshes-phosphorylation/curRadius_${radius}_axisymmetric_refined_${refinement}"
    done
done

partition=defq

# Diffusion, Refinements
for radius in 1 2 4 6 8 10
do
    for refinement in 0 1 2
    do
        for diffusion in 0.01 0.1 1.0 10.0 100.0
        do
            python3 main.py --partition=$partition  phosphorylation --diffusion $diffusion --curRadius $radius --mesh-folder "meshes-phosphorylation/curRadius_${radius}_refined_${refinement}"
        done
    done
done

#Same with axissymetric
for radius in 1 2 4 6 8 10
do
    for refinement in 0 1 2
    do
        for diffusion in 0.01 0.1 1.0 10.0 100.0
        do
            python3 main.py --partition=$partition  phosphorylation --axisymmetric --diffusion $diffusion --curRadius $radius --mesh-folder "meshes-phosphorylation/curRadius_${radius}_axisymmetric_refined_${refinement}"
        done
    done
done

# Run with MPI for D = 1, Raidus = 10 and 2 levels of refinement
for ntasks in 2 4 6 8 10 12 14 16
do
    python3 main.py --ntasks $ntasks --partition=$partition  phosphorylation --diffusion 1.0 --curRadius 10 --mesh-folder "meshes-phosphorylation/curRadius_10_refined_2"
done
