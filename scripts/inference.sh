#!/bin/bash
#SBATCH --job-name=reg-no
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
##SBATCH --mail-type=BEGIN
##SBATCH --mail-user=jy3694@nyu.edu

cat<<EOF
Job starts at: $(date)

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

singularity exec --nv --overlay /scratch/jy3694/data/face_occlusion.sqf:ro\
     --overlay /scratch/gm2724/vggface2hq.sqf:ro --overlay /scratch/jy3694/data/Celeb-DF-v2.sqf:ro \
     --overlay /scratch/jy3694/data/face_occlusion_generation.sqf:ro \
     --overlay /scratch/jy3694/torchenv.ext3:ro /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "source /ext3/env.sh;python inference.py --model_dir 8_1/regress_occlusion --use_retinafacenet --batch_size 128 --use_regression --lpips"