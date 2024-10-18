echo "Running nanopillar examples testing NPC stretch"
# Create meshes with symm considerations
# indentArray=\
# (0.0 0.2 0.4 0.6 0.8\
#  1.0 1.2 1.4 1.6 1.8\
#  2.0 2.2 2.4 2.6 2.8)
indentArray=\
(0.0 -0.2 -0.4\
 -0.6 -0.8 -1.0\
 -1.2 -1.4 -1.6)
for idx in 0 1 2 3 4 5 6 7;
do
    echo "Running simulation for u0 = ${u0Array[outer_idx]} and indentation=${indentArray[idx]}"
    python3 main.py --ntasks 1 --submit-tscc mechanotransduction \
    --mesh-folder /root/scratch/meshes/nanopillars_movenuc/nanopillars_movenuc${indentArray[idx]} \
    --time-step 0.01 --e-val 10000000 \
    --outdir /root/scratch/results_nanopillars_movenuc/nanopillars_movenuc${indentArray[idx]}\
    --reaction-rate-on-np 1 --curv-sens 1 --WASP-rate 0.01 \
    --a0-npc 5.0 --nuc-compression ${indentArray[idx]}
    sleep 100
done

echo "Done."
