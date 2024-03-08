# Create meshes
for shape in circle star rect
do
    for refinement in 0 1 2 3
    do
        python3 main.py mechanotransduction-preprocess --shape $shape --num-refinements $refinement --mesh-folder meshes-mechanotransduction/${shape}_hEdge_0.6_refined_${refinement}
    done
done

# Base case
for eval in 0.1 5.7 70000000
do
    for shape in circle star rect
    do
        for refinement in 0 1 2 3
        do
            python3 main.py mechanotransduction --mesh-folder meshes-mechanotransduction/${shape}_hEdge_0.6_refined_${refinement} --time-step 0.01 --e-val $eval --axisymmetric -o results-mechanotransduction/${shape}_hEdge_0.6_refined_${refinement}_e_${eval}
        done
    done
done

# Well mixed case
for eval in 0.1 5.7 70000000
do
    for shape in circle star rect
    do
        for refinement in 0 1 2 3
        do
            python3 main.py --dry-run mechanotransduction --mesh-folder meshes-mechanotransduction/${shape}_hEdge_0.6_refined_${refinement} --time-step 0.01 --e-val $eval --axisymmetric --well-mixed -o results-mechanotransduction/${shape}_hEdge_0.6_refined_${refinement}_e_${eval}_well-mixed
        done
    done
done