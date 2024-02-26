#!/bin/bash
#SBATCH --job-name  mechanotransduction-%N
#SBATCH --output slurm-%j.out-%N
#SBATCH --error slurm-%j.err-%N
#SBATCH --mail-type END 
#SBATCH --partition=condo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=100:00:00
#SBATCH --account=csd786
#SBATCH --qos=condo
#SBATCH --array=1-9

module load singularitypro/3.11
module load mpich/ge/gcc/64/3.4.2
cd /tscc/nfs/home/eafrancis/gitrepos/smart-comp-sci/scripts
echo $SLURM_ARRAY_TASK_ID
export IDX=$SLURM_ARRAY_TASK_ID
echo $IDX
mpirun -np 4 singularity exec --bind $HOME:/root/shared,\
$TMPDIR:/root/tmp,/tscc/lustre/ddn/scratch/eafrancis/mechanotransduction/smart-fixed:/root/scratch \
/tscc/nfs/home/eafrancis/smart-newmeshview.sif bash run_mechanotransduction2.sh
