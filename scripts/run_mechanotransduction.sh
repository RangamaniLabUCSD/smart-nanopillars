# Create meshes

# Axisymmetric
for refinement in 0 1 2 3
do
    for shape in circle star rect
    do
        echo "python3 main.py mechanotransduction-preprocess --shape $shape --num-refinements $refinement --mesh-folder meshes-mechanotransduction/${shape}_refined_${refinement}"
        python3 main.py mechanotransduction-preprocess --shape $shape --num-refinements $refinement --mesh-folder meshes-mechanotransduction/${shape}_refined_${refinement}
    done
done

# Full3D
for refinement in 0 1 2 3
do
    for shape in circle star rect
    do
        echo "python3 main.py mechanotransduction-preprocess --full-3d --shape $shape --num-refinements $refinement --mesh-folder meshes-mechanotransduction/${shape}_full3d_refined_${refinement}"
        python3 main.py mechanotransduction-preprocess --full-3d --shape $shape --num-refinements $refinement --mesh-folder meshes-mechanotransduction/${shape}_full3d_refined_${refinement}
    done
done


# 2D case
for well_mixed in 0 1
do
    for refinement in 0 1 2 3
    do
        for shape in circle star rect
        do
            for eval in 0.1 5.7 70000000
            do
                if [ $well_mixed -eq 1 ]
                then
                    python3 main.py mechanotransduction --mesh-folder meshes-mechanotransduction/${shape}_full3d_refined_${refinement} --time-step 0.01 --e-val $eval --axisymmetric --well-mixed
                else
                    python3 main.py mechanotransduction --mesh-folder meshes-mechanotransduction/${shape}_full3d_refined_${refinement} --time-step 0.01 --e-val $eval --axisymmetric
                fi
                sleep 10
            done
        done
    done
do


# 3D case
for well_mixed in 0 1
do
    for refinement in 0 1 2 3
    do
        for shape in circle star rect
        do
            for eval in 0.1 5.7 70000000
            do
                if [ $well_mixed -eq 1 ]
                then
                    python3 main.py mechanotransduction --mesh-folder meshes-mechanotransduction/${shape}_refined_${refinement} --time-step 0.01 --e-val $eval --axisymmetric --well-mixed
                else
                    python3 main.py mechanotransduction --mesh-folder meshes-mechanotransduction/${shape}_refined_${refinement} --time-step 0.01 --e-val $eval --axisymmetric
                fi
                sleep 10
            done
        done
    done
do

