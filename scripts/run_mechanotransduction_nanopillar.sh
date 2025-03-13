echo "Running nanopillar examples"
# Create meshes with symm considerations
radiusArray=\
(0.1 0.1 0.1 0.1 0.1 0.1\
 0.5 0.25 0.5 0.25\
 0.0 0.0 0.0 0.0 0.0)
pitchArray=\
(5.0 2.5 1.0 5.0 2.5 1.0\
 5.0 2.5 5.0 2.5\
 0.0 0.0 0.0 0.0 0.0)
heightArray=\
(1.0 1.0 1.0 3.0 3.0 3.0\
 1.0 1.0 3.0 3.0\
 0.0 0.0 0.0 0.0 0.0)
cellRadArray=\
(20.25 18.52 16.55 19.93 18.04 15.39\
 20.01 17.45 18.06 17.64\
 22.48 18.08 15.39 14.18 12.33)
EModArray=\
(10000000 10000000 10000000 10000000 10000000\
 10000000 10000000 10000000 10000000 10000000\
 10000000 14 7 3 1)
curv0Array=(0 0 2 5 10 0 2 5 10 0 2 5 10)
waspArray=(0 0.001 0.001 0.001 0.001 0.01 0.01 0.01 0.01 0.1 0.1 0.1 0.1)
# nprateArray=(1 1 1 1 1 0.3 0)
for outeridx in 0 5 6 7 8;
do
    for idx in 0 1 2 5 6 7 ;
    do
        echo "Running simulation for h${heightArray[idx]}_p${pitchArray[idx]}_r${radiusArray[idx]} nanopillars for cellRad=${cellRadArray[idx]} and EMod=${EModArray[idx]}"
        python3 main.py --ntasks 1 --submit-tscc mechanotransduction \
        --mesh-folder /root/scratch/meshes/nanopillars_finalCalcCoarse/nanopillars_h${heightArray[idx]}_p${pitchArray[idx]}_r${radiusArray[idx]}_cellRad${cellRadArray[idx]} \
        --time-step 0.01 --e-val ${EModArray[idx]} \
        --outdir /root/scratch/results_nanopillars_withWASP/nanopillars_h${heightArray[idx]}_p${pitchArray[idx]}_r${radiusArray[idx]}_cellRad${cellRadArray[idx]}_nprate${nprateArray[outeridx]}_curvSens${curv0Array[outeridx]}_wasp${waspArray[outeridx]} \
        --reaction-rate-on-np 1.0 --curv-sens ${curv0Array[outeridx]} --WASP-rate ${waspArray[outeridx]} --alt-yap-diffusion
        sleep 30
    done
done

for outeridx in 0 5;
do
    for idx in 10 11 12 13 14;
    do
        echo "Running simulation for h${heightArray[idx]}_p${pitchArray[idx]}_r${radiusArray[idx]} nanopillars for cellRad=${cellRadArray[idx]} and EMod=${EModArray[idx]}"
        python3 main.py --ntasks 1 --submit-tscc mechanotransduction \
        --mesh-folder /root/scratch/meshes/nanopillars_finalCalcCoarse/nanopillars_h${heightArray[idx]}_p${pitchArray[idx]}_r${radiusArray[idx]}_cellRad${cellRadArray[idx]} \
        --time-step 0.01 --e-val ${EModArray[idx]} --WASP-rate ${waspArray[outeridx]} \
        --outdir /root/scratch/results_nanopillars_withWASP/nanopillars_h${heightArray[idx]}_p${pitchArray[idx]}_r${radiusArray[idx]}_cellRad${cellRadArray[idx]}_wasp${waspArray[outeridx]} --alt-yap-diffusion
        sleep 30
    done
done

echo "Done."
