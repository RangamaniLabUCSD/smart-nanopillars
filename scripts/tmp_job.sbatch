#!/bin/bash
#SBATCH --job-name="mechanotransduction"
#SBATCH --partition=defq
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt

module purge
module use /cm/shared/ex3-modules/latest/modulefiles
module load slurm
module load  gcc-10.1.0
module load libgfortran-5.0.0
. /home/henriknf/local/src/spack/share/spack/setup-env.sh
spack env activate fenics-dev-fpqga

SCRATCH_DIRECTORY=/global/D1/homes/${USER}/smart-comp-sci/mechanotransduction/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
echo "Scratch directory: ${SCRATCH_DIRECTORY}"

echo 'Run command: python /root/shared/gitrepos/smart-comp-sci/mechanotransduction-example/mechanotransduction.py --outdir "${SCRATCH_DIRECTORY}" --mesh-folder /root/scratch/meshes/circle_hEdge_0.3_full3dfalse --time-step 0.01 --e-val 0.1 --z-cutoff 0.0001 --axisymmetric'
mpirun -n 1 python /root/shared/gitrepos/smart-comp-sci/mechanotransduction-example/mechanotransduction.py --outdir "${SCRATCH_DIRECTORY}" --mesh-folder /root/scratch/meshes/circle_hEdge_0.3_full3dfalse --time-step 0.01 --e-val 0.1 --z-cutoff 0.0001 --axisymmetric
# Move log file to results folder
mv ${SLURM_JOBID}-* ${SCRATCH_DIRECTORY}
