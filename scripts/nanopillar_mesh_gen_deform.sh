echo "Saving meshes for nanopillar examples"
# Create meshes with symm considerations
indentArray=\
(0.0 0.2 0.4 0.6 0.8\
 1.0 1.2 1.4 1.6 1.8\
 2.0 2.2 2.4 2.6 2.8)
for idx in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14;
do
    echo "Writing mesh for for indentation=${indentArray[idx]}"
    python3 main.py mechanotransduction-preprocess --shape circle --hEdge 0.4 --hInnerEdge 0.4 \
--mesh-folder /root/shared/gitrepos/smart-comp-sci-data/meshes/nanopillars_new_hNPRefined_coarser/nanopillars_indent${indentArray[idx]} \
--contact-rad 15.5 \
--nanopillar-radius 0.5 --nanopillar-height 3.0 --nanopillar-spacing 3.5 \
--nuc-compression ${indentArray[idx]}
done
echo "Done."