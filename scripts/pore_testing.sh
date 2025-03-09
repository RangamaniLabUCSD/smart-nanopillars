echo "Running nanopillar examples testing nuclear rupture"
# Create meshes with symm considerations
indentArray=(0.8 1.8 2.8)
# poreLocArray=(0 0 0 3.5 3.5 3.5)
poreSizeArray=(0.1 0.2 0.5)
# poreRateArray=(1 100)
transportRateArray=(100000 300000)
# transportRatioArray=(1 3 10 1 3 10 1 3 10 1 3 10)
# for indentIdx in 0 1 2;
# do
#     for poreIdx in 0 1 2;
#     do
#         echo "Running simulation for pore testing with indentation=${indentArray[indentIdx]}"
#         python3 main.py --submit-tscc mechanotransduction-nuc-only \
#         --mesh-folder /root/shared/gitrepos/smart-nanopillars/meshes/nanopillars_indent/nanopillars_indent${indentArray[indentIdx]} --time-step 0.01 \
#         --outdir /root/scratch/results_nanopillars_indentation_poreTesting_noPhosNewRatio \
#         --full-sims-folder /root/scratch/results_nanopillars_indentation \
#         --pore-loc 0.0 --pore-size ${poreSizeArray[poreIdx]} \
#         --a0-npc 5.0 --nuc-compression ${indentArray[indentIdx]}
#         sleep 500
#     done
# done

for indentIdx in 0 1 2;
do
    for poreIdx in 0 1 2;
    do
        for rateIdx in 0 1 2;
        do
            echo "Running simulation for pore testing with indentation=${indentArray[indentIdx]}, alt diffusion"
            python3 main.py --submit-tscc mechanotransduction-nuc-only \
            --mesh-folder /root/shared/gitrepos/smart-nanopillars/meshes/nanopillars_indent/nanopillars_indent${indentArray[indentIdx]} --time-step 0.01 \
            --outdir /root/scratch/results_nanopillars_indentation_poreTesting_noPhosNewRatio_altDiffusion \
            --full-sims-folder /root/scratch/results_nanopillars_indentation_altDiffusion \
            --transport-rate ${transportRateArray[rateIdx]} \
            --pore-loc 0.0 --pore-size ${poreSizeArray[poreIdx]} \
            --a0-npc 5.0 --nuc-compression ${indentArray[indentIdx]} --alt-yap-diffusion
            sleep 500
        done
    done
done

echo "Done."