if [ $IDX -gt 9 ]
then
    full3d=true
else
    full3d=false
fi
idx=$(((IDX-1) % 9 + 1))
echo $idx
# submit jobs by IDX
case $idx in
    # circle sims
    1)
        echo "Running case 1, full3d=$full3d"
        python3 main.py mechanotransduction --mesh-folder /root/scratch/meshes/circle_hEdge_0.3_full3d$full3d --time-step 0.01 --e-val 0.1 --axisymmetric --outdir /root/scratch/results/circle_E0.1_full3d$full3d
        ;;
    2)
        echo "Running case 2, full3d=$full3d"
        python3 main.py mechanotransduction --mesh-folder /root/scratch/meshes/circle_hEdge_0.3_full3d$full3d --time-step 0.01 --e-val 5.7 --axisymmetric --outdir /root/scratch/results/circle_E5.7_full3d$full3d
        ;;
    3)
        echo "Running case 3, full3d=$full3d"
        python3 main.py mechanotransduction --mesh-folder /root/scratch/meshes/circle_hEdge_0.3_full3d$full3d --time-step 0.01 --e-val 70000000 --axisymmetric --outdir /root/scratch/results/circle_E70000000_full3d$full3d
        ;;
    # rect sims
    4)
        echo "Running case 4, full3d=$full3d"
        python3 main.py mechanotransduction --mesh-folder /root/scratch/meshes/rect_hEdge_0.3_full3d$full3d --time-step 0.01 --e-val 0.1 --outdir /root/scratch/results/rect_E0.1_full3d$full3d
        ;;
    5)
        echo "Running case 5, full3d=$full3d"
        python3 main.py mechanotransduction --mesh-folder /root/scratch/meshes/rect_hEdge_0.3_full3d$full3d --time-step 0.01 --e-val 5.7 --outdir /root/scratch/results/rect_E5.7_full3d$full3d
        ;;
    6)
        echo "Running case 6, full3d=$full3d"
        python3 main.py mechanotransduction --mesh-folder /root/scratch/meshes/rect_hEdge_0.3_full3d$full3d --time-step 0.01 --e-val 70000000 --outdir /root/scratch/results/rect_E70000000_full3d$full3d
        ;;
    # star sims
    7)
        echo "Running case 7, full3d=$full3d"
        python3 main.py mechanotransduction --mesh-folder /root/scratch/meshes/star_hEdge_0.3_full3d$full3d --time-step 0.01 --e-val 0.1 --outdir /root/scratch/results/star_E0.1_full3d$full3d
        ;;
    8)
        echo "Running case 8, full3d=$full3d"
        python3 main.py mechanotransduction --mesh-folder /root/scratch/meshes/star_hEdge_0.3_full3d$full3d --time-step 0.01 --e-val 5.7 --outdir /root/scratch/results/star_E5.7_full3d$full3d
        ;;
    9)
        echo "Running case 9, full3d=$full3d"
        python3 main.py mechanotransduction --mesh-folder /root/scratch/meshes/star_hEdge_0.3_full3d$full3d --time-step 0.01 --e-val 70000000 --outdir /root/scratch/results/star_E70000000_full3d$full3d
        ;;
esac
