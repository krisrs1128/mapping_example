#!/bin/bash
#SBATCH --job-name=mapping
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --account=def-bengioy
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output /home/sankarak/logs/mapping-%j.out
#SBATCH --error /home/sankarak/logs/mapping-%j.err

rsync -avz /scratch/mtrazzak/datasets/lcm/ $SLURM_TMPDIR
module load singularity
singularity shell --nv --bind $SLURM_TMPDIR:/data/,$HOME/mapping_example:/home/mapping_example,$SCRATCH/:$SCRATCH/ $SCRATCH/images/mapping.img /home/mapping_example/cluster/mapping.sh
