echo "Saving meshes for nanopillar examples"
# Create meshes with symm considerations
indentArray=\
(0.0 -0.2 -0.4\
 -0.6 -0.8 -1.0\
 -1.2 -1.4 -1.6)
for idx in 8 7 6 5 4 3 2 1 0;
do
    echo "Writing mesh for for indentation=${indentArray[idx]}"
    python3 main.py mechanotransduction-preprocess --hEdge 0.5 --hInnerEdge 0.5 \
--mesh-folder /root/shared/gitrepos/smart-comp-sci-data/meshes/nanopillars_movenuc/nanopillars_movenuc${indentArray[idx]} \
--contact-rad 17.45 \
--nanopillar-radius 0.25 --nanopillar-height 1.0 --nanopillar-spacing 2.5 \
--nuc-compression ${indentArray[idx]}
done
echo "Done."