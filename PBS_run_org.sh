#!/bin/csh
#PBS -N gpu
#PBS -j oe
#PBS -q GPU-1
#PBS -lselect=1:ngpus=1


source /etc/profile.d/modules.csh
module purge
module load singularity


cd /home/nguyennd/work/lstm
Queue Name
Resources
singularity exec --nv /work/opt/container_images/tensorflow_20.03-tf2-py3.sif 
python sequential_model.py

#PBS -M nguyennd@jaist.ac.jp -m be