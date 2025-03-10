echo "Running nanopillar examples testing NPC stretch"
indentArray=\
(0.0 0.4 0.8\
 1.0 1.4 1.8\
 2.0 2.4 2.8)
a0Array=(0 2.5 5.0 7.5 10.0)
for outer_idx in 0 1 2 3; # 4 5 6;
do
    for idx in 0 1 2 3 4 5 6 7 8;
    do
        echo "Running simulation for u0 = ${u0Array[outer_idx]} and indentation=${indentArray[idx]}"
        python3 main.py --submit-tscc mechanotransduction \
        --mesh-folder /root/shared/gitrepos/smart-nanopillars/meshes/nanopillars_indent/nanopillars_indent${indentArray[idx]} \
        --time-step 0.01 --e-val 10000000 \
        --outdir /root/scratch/results_nanopillars_indentation_altDiffusion/nanopillars_indent${indentArray[idx]}_a0_${a0Array[outer_idx]}\
        --reaction-rate-on-np 1 --curv-sens 5 --WASP-rate 0.01 \
        --a0-npc ${a0Array[outer_idx]} --nuc-compression ${indentArray[idx]} --alt-yap-diffusion
        sleep 100
    done
done

echo "Done."
