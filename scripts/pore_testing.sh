echo "Running nanopillar examples testing nuclear rupture"
indentArray=(0.8 1.8 2.8)
poreSizeArray=(0.1 0.2 0.1 0.2 0.1 0.2)
transportRateArray=(3000 10000 30000 100000)
transportRatioArray=(1 1 2 2 5 5)

for indentIdx in 0 1 2;
do
    for poreIdx in 0 1 2 3 4 5;
    do
        for rateIdx in 0 1 2 3;
        do
            echo "Running simulation for pore testing with indentation=${indentArray[indentIdx]}, alt diffusion"
            python3 main.py --submit-tscc mechanotransduction-nuc-only \
            --mesh-folder /root/shared/gitrepos/smart-nanopillars/meshes/nanopillars_indent/nanopillars_indent${indentArray[indentIdx]} --time-step 0.01 \
            --outdir /root/scratch/results_nanopillars_indentation_poreTesting_noPhosNewRatio_altDiffusionREDO \
            --full-sims-folder /root/scratch/results_nanopillars_indentation_altDiffusionREDO \
            --transport-rate ${transportRateArray[rateIdx]} --transport-ratio ${transportRatioArray[poreIdx]} \
            --pore-loc 0.0 --pore-size ${poreSizeArray[poreIdx]} \
            --a0-npc 5.0 --nuc-compression ${indentArray[indentIdx]} --alt-yap-diffusion
            sleep 100
        done
    done
done

echo "Done."