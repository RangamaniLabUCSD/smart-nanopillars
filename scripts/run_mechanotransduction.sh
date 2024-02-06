# Create meshes
# python3 main.py preprocess-mech-mesh --shape circle --num-refinements 0 --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_0
# python3 main.py preprocess-mech-mesh --shape circle --num-refinements 1 --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_1
# python3 main.py preprocess-mech-mesh --shape circle --num-refinements 2 --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_2
# python3 main.py preprocess-mech-mesh --shape circle --num-refinements 3 --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_3

# Base case
# python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_0 --time-step 0.01 --e-val 70000000 --axisymmetric

# Run spatial convergence
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_1 --time-step 0.01 --e-val 70000000 --axisymmetric
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_2 --time-step 0.01 --e-val 70000000 --axisymmetric
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_3 --time-step 0.01 --e-val 70000000 --axisymmetric

# # Run temporal convergence
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_0 --time-step 0.02 --e-val 70000000 --axisymmetric
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_0 --time-step 0.005 --e-val 70000000 --axisymmetric
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_0 --time-step 0.001 --e-val 70000000 --axisymmetric

# # Test mass conservation
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6_refined_0 --time-step 0.01 --e-val 70000000 --axisymmetric --no-enforce-mass-conservation