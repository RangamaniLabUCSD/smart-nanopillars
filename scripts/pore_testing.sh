echo "Running nanopillar examples testing nuclear rupture"
# Create meshes with symm considerations
indentArray=(0.8 1.8 2.8)
poreLocArray=(0 0 0 3.5 3.5 3.5)
poreSizeArray=(0.1 0.2 0.5 0.1 0.2 0.5)
# poreRateArray=(1 100)
# transportRateArray=(10 10 10 100 100 100)
# transportRatioArray=(1 3 10 1 3 10)

for indentIdx in 0 1 2;
do
    for poreIdx in 0 1 2 3 4 5;
    do
        echo "Running simulation for pore testing with indentation=${indentArray[indentIdx]}"
        python3 main.py --ntasks 1 --submit-tscc mechanotransduction-nuc-only \
        --mesh-folder /root/scratch/meshes/nanopillars_new/nanopillars_indent${indentArray[indentIdx]} --time-step 0.01 \
        --outdir /root/scratch/results_nanopillars_indentation_poreTesting_REDOAGAIN \
        --full-sims-folder /root/scratch/results_nanopillars_indentation_newStretchCombined \
        --pore-loc ${poreLocArray[poreIdx]} --pore-size ${poreSizeArray[poreIdx]} \
        --u0-npc 5.0 --nuc-compression ${indentArray[indentIdx]}
        sleep 60
    done
done

echo "Done."