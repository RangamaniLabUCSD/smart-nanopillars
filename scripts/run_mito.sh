echo $IDX
# submit jobs by IDX
case $IDX in
    1)
        echo "Running case 1, curv dep = -20"
        python3 main.py mito --mesh-file /root/scratch/meshes/mito_mesh.h5  --curv-file /root/scratch/mito_curv.xdmf --outdir /root/scratch/sweep_figure_redo/results_curvdepneg20 --curv-dep=-20
        ;;
    2)
        echo "Running case 2, curv dep = -10"
        python3 main.py mito --mesh-file /root/scratch/meshes/mito_mesh.h5  --curv-file /root/scratch/mito_curv.xdmf --outdir /root/scratch/sweep_figure_redo/results_curvdepneg10 --curv-dep=-10
        ;;
    3)
        echo "Running case 3, curv dep = 0"
        python3 main.py mito --mesh-file /root/scratch/meshes/mito_mesh.h5  --curv-file /root/scratch/mito_curv.xdmf --outdir /root/scratch/sweep_figure_redo/results_curvdep0 --curv-dep=0
        ;;
    4)
        echo "Running case 4, curv dep = 10"
        python3 main.py mito --mesh-file /root/scratch/meshes/mito_mesh.h5  --curv-file /root/scratch/mito_curv.xdmf --outdir /root/scratch/sweep_figure_redo/results_curvdep10 --curv-dep=10
        ;;
    5)
        echo "Running case 5, curv dep = 20"
        python3 main.py mito --mesh-file /root/scratch/meshes/mito_mesh.h5  --curv-file /root/scratch/mito_curv.xdmf --outdir /root/scratch/sweep_figure_redo/results_curvdep20 --curv-dep=20
        ;;
esac