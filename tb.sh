#!/bin/bash

#SBATCH --job-name=tensorboard
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4GB
#SBATCH --time=48:00:00
##SBATCH --mail-type=BEGIN
##SBATCH --mail-user=jy3694@nyu.edu

#port=8123
port=$(shuf -i 6000-9999 -n 1)
/usr/bin/ssh -N -f -R $port:localhost:$port log-1
/usr/bin/ssh -N -f -R $port:localhost:$port log-2
/usr/bin/ssh -N -f -R $port:localhost:$port log-3

cat<<EOF
Job starts at: $(date)

ssh -L $port:localhost:$port jy3694@greene.hpc.nyu.edu 
EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi
dir=$1


singularity exec --overlay /scratch/jy3694/venv_tf.ext3:ro \
	/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	/bin/bash -c "source /ext3/env.sh; tensorboard --logdir=$dir --reload_multifile True --port=$port --host=localhost "

echo "tensorboard --logdir=$dir --port=$port --host=localhost"