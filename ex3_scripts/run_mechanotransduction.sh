# Create meshes
# python3 main.py preprocess-mech-mesh --shape circle --hEdge 0.8 --hInnerEdge 0.8 --mesh-folder meshes-mechanotransduction/circle_hEdge_0.8
# python3 main.py preprocess-mech-mesh --shape circle --hEdge 0.6 --hInnerEdge 0.6 --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6
# python3 main.py preprocess-mech-mesh --shape circle --hEdge 0.4 --hInnerEdge 0.4 --mesh-folder meshes-mechanotransduction/circle_hEdge_0.4
# python3 main.py preprocess-mech-mesh --shape circle --hEdge 0.2 --hInnerEdge 0.2 --mesh-folder meshes-mechanotransduction/circle_hEdge_0.2

# # # Base case
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6 --time-step 0.01 --e-val 70000000 --axisymmetric

# Run spatial convergence
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.8 --time-step 0.01 --e-val 70000000 --axisymmetric
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.4 --time-step 0.01 --e-val 70000000 --axisymmetric
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.2 --time-step 0.01 --e-val 70000000 --axisymmetric

# # Run temporal convergence
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6 --time-step 0.02 --e-val 70000000 --axisymmetric
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6 --time-step 0.005 --e-val 70000000 --axisymmetric
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6 --time-step 0.001 --e-val 70000000 --axisymmetric

# # Test mass conservation
python3 main.py --submit-ex3 mechanotransduction --mesh-folder meshes-mechanotransduction/circle_hEdge_0.6 --time-step 0.01 --e-val 70000000 --axisymmetric --no-enforce-mass-conservation