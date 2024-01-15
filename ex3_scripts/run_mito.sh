python3 main.py preprocess-mito-mesh --input-mesh-file /root/shared/gitrepos/smart-comp-sci/meshes/mito1_mesh.xml --output-mesh-file meshes/mito_mesh.h5 --input-curv-file /root/shared/gitrepos/smart-comp-sci/meshes/mito1_curvature.xml --output-curv-file meshes/mito_curv.xdmf
python3 main.py mito --mesh-file meshes/mito_mesh.h5  --curv-file meshes/mito_curv.xdmf --outdir /root/scratch/eafrancis/mito/sweep_gamer_curv/results_curvdepneg40 --curv-dep=-40 --enforce-mass-conservation &
sleep 10
python3 main.py mito --mesh-file meshes/mito_mesh.h5  --curv-file meshes/mito_curv.xdmf --outdir /root/scratch/eafrancis/mito/sweep_gamer_curv/results_curvdepneg20 --curv-dep=-20 --enforce-mass-conservation &
sleep 10
python3 main.py mito --mesh-file meshes/mito_mesh.h5  --curv-file meshes/mito_curv.xdmf --outdir /root/scratch/eafrancis/mito/sweep_gamer_curv/results_curvdepneg10 --curv-dep=-10 --enforce-mass-conservation &
sleep 10
python3 main.py mito --mesh-file meshes/mito_mesh.h5  --curv-file meshes/mito_curv.xdmf --outdir /root/scratch/eafrancis/mito/sweep_gamer_curv/results_curvdep10 --curv-dep 10 --enforce-mass-conservation &
sleep 10
python3 main.py mito --mesh-file meshes/mito_mesh.h5  --curv-file meshes/mito_curv.xdmf --outdir /root/scratch/eafrancis/mito/sweep_gamer_curv/results_curvdep20 --curv-dep 20 --enforce-mass-conservation &
sleep 10
python3 main.py mito --mesh-file meshes/mito_mesh.h5  --curv-file meshes/mito_curv.xdmf --outdir /root/scratch/eafrancis/mito/sweep_gamer_curv/results_curvdep40 --curv-dep 40 --enforce-mass-conservation