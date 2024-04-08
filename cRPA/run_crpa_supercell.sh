#!/bin/bash
#SBATCH --account=berkelbach
#SBATCH --job-name hbn_c2_6_7_dzvp
#SBATCH --time=48:30:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=700000MB 
#SBATCH --array=6-7

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=sjb2225@columbia.edu

JOBNAME=$SLURM_JOB_NAME

. /burg/berkelbach/users/sjb2225/build/spack/share/spack/setup-env.sh
spack env activate -p pyscf
export PYTHONPATH=/burg/berkelbach/users/sjb2225/build/pyscf:$PYTHONPATH
export PYSCF_MAX_MEMORY=690000
export OMP_NUM_THREADS=32

arr=(${SLURM_ARRAY_TASK_ID})
echo "$(python crpa_supercell.py ${arr[@]})" >> "hbn_c2_dzvp_"${arr[0]}".txt"
