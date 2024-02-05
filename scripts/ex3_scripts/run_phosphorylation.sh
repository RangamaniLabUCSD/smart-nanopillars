# # Create meshes
# python3 main.py preprocess-phosphorylation-mesh --curRadius 1.0 --num-refinements 0 --mesh-folder meshes-phosphorylation/curRadius_1_refined_0
# python3 main.py preprocess-phosphorylation-mesh --curRadius 2.0 --num-refinements 0 --mesh-folder meshes-phosphorylation/curRadius_2_refined_0
# python3 main.py preprocess-phosphorylation-mesh --curRadius 4.0 --num-refinements 0 --mesh-folder meshes-phosphorylation/curRadius_4_refined_0
# python3 main.py preprocess-phosphorylation-mesh --curRadius 6.0 --num-refinements 0 --mesh-folder meshes-phosphorylation/curRadius_6_refined_0
# python3 main.py preprocess-phosphorylation-mesh --curRadius 8.0 --num-refinements 0 --mesh-folder meshes-phosphorylation/curRadius_8_refined_0
# python3 main.py preprocess-phosphorylation-mesh --curRadius 10.0 --num-refinements 0 --mesh-folder meshes-phosphorylation/curRadius_10_refined_0

# python3 main.py preprocess-phosphorylation-mesh --curRadius 1.0 --num-refinements 1 --mesh-folder meshes-phosphorylation/curRadius_1_refined_1
# python3 main.py preprocess-phosphorylation-mesh --curRadius 2.0 --num-refinements 1 --mesh-folder meshes-phosphorylation/curRadius_2_refined_1
# python3 main.py preprocess-phosphorylation-mesh --curRadius 4.0 --num-refinements 1 --mesh-folder meshes-phosphorylation/curRadius_4_refined_1
# python3 main.py preprocess-phosphorylation-mesh --curRadius 6.0 --num-refinements 1 --mesh-folder meshes-phosphorylation/curRadius_6_refined_1
# python3 main.py preprocess-phosphorylation-mesh --curRadius 8.0 --num-refinements 1 --mesh-folder meshes-phosphorylation/curRadius_8_refined_1
# python3 main.py preprocess-phosphorylation-mesh --curRadius 10.0 --num-refinements 1 --mesh-folder meshes-phosphorylation/curRadius_10_refined_1

# python3 main.py preprocess-phosphorylation-mesh --curRadius 1.0 --num-refinements 2 --mesh-folder meshes-phosphorylation/curRadius_1_refined_2
# python3 main.py preprocess-phosphorylation-mesh --curRadius 2.0 --num-refinements 2 --mesh-folder meshes-phosphorylation/curRadius_2_refined_2
# python3 main.py preprocess-phosphorylation-mesh --curRadius 4.0 --num-refinements 2 --mesh-folder meshes-phosphorylation/curRadius_4_refined_2
# python3 main.py preprocess-phosphorylation-mesh --curRadius 6.0 --num-refinements 2 --mesh-folder meshes-phosphorylation/curRadius_6_refined_2
# python3 main.py preprocess-phosphorylation-mesh --curRadius 8.0 --num-refinements 2 --mesh-folder meshes-phosphorylation/curRadius_8_refined_2
# python3 main.py preprocess-phosphorylation-mesh --curRadius 10.0 --num-refinements 2 --mesh-folder meshes-phosphorylation/curRadius_10_refined_2

# Basic case
python3 main.py --submit-ex3 phosphorylation --curRadius 1.0 --mesh-folder meshes-phosphorylation/curRadius_1_refined_0 
python3 main.py --submit-ex3 phosphorylation --curRadius 2.0 --mesh-folder meshes-phosphorylation/curRadius_2_refined_0 
python3 main.py --submit-ex3 phosphorylation --curRadius 4.0 --mesh-folder meshes-phosphorylation/curRadius_4_refined_0 
python3 main.py --submit-ex3 phosphorylation --curRadius 6.0 --mesh-folder meshes-phosphorylation/curRadius_6_refined_0 
python3 main.py --submit-ex3 phosphorylation --curRadius 8.0 --mesh-folder meshes-phosphorylation/curRadius_8_refined_0 
python3 main.py --submit-ex3 phosphorylation --curRadius 10.0 --mesh-folder meshes-phosphorylation/curRadius_10_refined_0 

# dt = 0.005
python3 main.py --submit-ex3 phosphorylation --time-step 0.005 --curRadius 1.0 --mesh-folder meshes-phosphorylation/curRadius_1_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.005 --curRadius 2.0 --mesh-folder meshes-phosphorylation/curRadius_2_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.005 --curRadius 4.0 --mesh-folder meshes-phosphorylation/curRadius_4_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.005 --curRadius 6.0 --mesh-folder meshes-phosphorylation/curRadius_6_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.005 --curRadius 8.0 --mesh-folder meshes-phosphorylation/curRadius_8_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.005 --curRadius 10.0 --mesh-folder meshes-phosphorylation/curRadius_10_refined_0

# dt = 0.0025
python3 main.py --submit-ex3 phosphorylation --time-step 0.0025 --curRadius 1.0 --mesh-folder meshes-phosphorylation/curRadius_1_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.0025 --curRadius 2.0 --mesh-folder meshes-phosphorylation/curRadius_2_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.0025 --curRadius 4.0 --mesh-folder meshes-phosphorylation/curRadius_4_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.0025 --curRadius 6.0 --mesh-folder meshes-phosphorylation/curRadius_6_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.0025 --curRadius 8.0 --mesh-folder meshes-phosphorylation/curRadius_8_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.0025 --curRadius 10.0 --mesh-folder meshes-phosphorylation/curRadius_10_refined_0

# dt = 0.001
python3 main.py --submit-ex3 phosphorylation --time-step 0.001 --curRadius 1.0 --mesh-folder meshes-phosphorylation/curRadius_1_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.001 --curRadius 2.0 --mesh-folder meshes-phosphorylation/curRadius_2_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.001 --curRadius 4.0 --mesh-folder meshes-phosphorylation/curRadius_4_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.001 --curRadius 6.0 --mesh-folder meshes-phosphorylation/curRadius_6_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.001 --curRadius 8.0 --mesh-folder meshes-phosphorylation/curRadius_8_refined_0
python3 main.py --submit-ex3 phosphorylation --time-step 0.001 --curRadius 10.0 --mesh-folder meshes-phosphorylation/curRadius_10_refined_0

# Refinement 1
python3 main.py --submit-ex3 phosphorylation --curRadius 1.0 --mesh-folder meshes-phosphorylation/curRadius_1_refined_1 
python3 main.py --submit-ex3 phosphorylation --curRadius 2.0 --mesh-folder meshes-phosphorylation/curRadius_2_refined_1 
python3 main.py --submit-ex3 phosphorylation --curRadius 4.0 --mesh-folder meshes-phosphorylation/curRadius_4_refined_1 
python3 main.py --submit-ex3 phosphorylation --curRadius 6.0 --mesh-folder meshes-phosphorylation/curRadius_6_refined_1 
python3 main.py --submit-ex3 phosphorylation --curRadius 8.0 --mesh-folder meshes-phosphorylation/curRadius_8_refined_1 
python3 main.py --submit-ex3 phosphorylation --curRadius 10.0 --mesh-folder meshes-phosphorylation/curRadius_10_refined_1 

# Refinement 2
python3 main.py --submit-ex3 phosphorylation --curRadius 1.0 --mesh-folder meshes-phosphorylation/curRadius_1_refined_2 
python3 main.py --submit-ex3 phosphorylation --curRadius 2.0 --mesh-folder meshes-phosphorylation/curRadius_2_refined_2 
python3 main.py --submit-ex3 phosphorylation --curRadius 4.0 --mesh-folder meshes-phosphorylation/curRadius_4_refined_2 
python3 main.py --submit-ex3 phosphorylation --curRadius 6.0 --mesh-folder meshes-phosphorylation/curRadius_6_refined_2 
python3 main.py --submit-ex3 phosphorylation --curRadius 8.0 --mesh-folder meshes-phosphorylation/curRadius_8_refined_2 
python3 main.py --submit-ex3 phosphorylation --curRadius 10.0 --mesh-folder meshes-phosphorylation/curRadius_10_refined_2 