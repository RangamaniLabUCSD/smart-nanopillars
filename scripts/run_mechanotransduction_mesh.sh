# Create meshes with symm considerations
python3 main.py mechanotransduction-preprocess --shape circle --hEdge 0.2 --hInnerEdge 0.2 --mesh-folder /root/scratch/meshes/circle_hEdge_0.2_full3dfalse
python3 main.py mechanotransduction-preprocess --shape rect --hEdge 0.2 --hInnerEdge 0.2 --mesh-folder /root/scratch/meshes/rect_hEdge_0.2_full3dfalse
python3 main.py mechanotransduction-preprocess --shape star --hEdge 0.2 --hInnerEdge 0.2 --mesh-folder /root/scratch/meshes/star_hEdge_0.2_full3dfalse
# Create full 3d meshes
python3 main.py mechanotransduction-preprocess --shape circle --hEdge 0.2 --hInnerEdge 0.2 --mesh-folder /root/scratch/meshes/circle_hEdge_0.2_full3dtrue --full-3d
python3 main.py mechanotransduction-preprocess --shape rect --hEdge 0.2 --hInnerEdge 0.2 --mesh-folder /root/scratch/meshes/rect_hEdge_0.2_full3dtrue --full-3d
python3 main.py mechanotransduction-preprocess --shape star --hEdge 0.2 --hInnerEdge 0.2 --mesh-folder /root/scratch/meshes/star_hEdge_0.2_full3dtrue --full-3d
