# Create meshes

for refinement in 0 1 2 3
do
    echo "python3 main.py mechanotransduction-preprocess --shape circle --num-refinements $refinement --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_${refinement}"
    python3 main.py mechanotransduction-preprocess --shape circle --num-refinements $refinement --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_${refinement}
done

# Create meshes
for shape in star rect
do
    echo "python3 main.py mechanotransduction-preprocess --shape $shape --num-refinements 0 --mesh-folder meshes-mechanotransduction/${shape}_hEdge_0.6_refined_0"
    python3 main.py mechanotransduction-preprocess --shape $shape --num-refinements 0 --mesh-folder meshes-mechanotransduction/${shape}_hEdge_0.6_refined_0  
done


# Base case
for eval in 0.1 5.7 70000000
do
    for refinement in 0 1 2 3
    do
        python3 main.py mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_${refinement} --time-step 0.01 --e-val $eval --axisymmetric -o results-mechanotransduction/circle_hEdge_0.6_refined_${refinement}_e_${eval}
    done
done

for eval in 0.1 5.7 70000000
do
    for shape in star rect
    do
        python3 main.py mechanotransduction --mesh-folder meshes-mechanotransduction/${shape}_hEdge_0.6_refined_0 --time-step 0.01 --e-val $eval -o results-mechanotransduction/${shape}_hEdge_0.6_refined_0_e_${eval}
    done
done

# Well mixed case
for eval in 0.1 5.7 70000000
do
    for refinement in 0 1 2 3
    do
        python3 main.py --dry-run mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_${refinement} --time-step 0.01 --e-val $eval --axisymmetric --well-mixed -o results-mechanotransduction/circle_hEdge_0.6_refined_${refinement}_e_${eval}_well-mixed
    done
done

for eval in 0.1 5.7 70000000
do
    for shape in star rect
    do
        python3 main.py --dry-run mechanotransduction --mesh-folder meshes-mechanotransduction/${shape}_hEdge_0.6_refined_0 --time-step 0.01 --e-val $eval --well-mixed -o results-mechanotransduction/${shape}_hEdge_0.6_refined_0_e_${eval}_well-mixed
    done
done