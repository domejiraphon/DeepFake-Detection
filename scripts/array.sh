#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-10
##SBATCH --partition=rtx8000,v100
#SBATCH --time=8:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=jy3694@nyu.edu

module purge
cd /home/gm2724/vast/anomaly_detector

singularity exec --nv --overlay /scratch/jy3694/data/face_occlusion.sqf:ro\
     --overlay /scratch/gm2724/vggface2hq.sqf:ro --overlay /scratch/jy3694/data/Celeb-DF-v2.sqf:ro \
     --overlay /scratch/jy3694/data/face_occlusion_generation.sqf:ro \
     --overlay /scratch/jy3694/torchenv.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
     /bin/bash -c "source /ext3/env.sh;python run_score.py $SLURM_ARRAY_TASK_ID"

