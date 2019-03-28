#!/bin/bash -l
#PBS -l walltime=00:01:00,nodes=1:ppn=24:gpus=2,mem=125gb
#PBS -m abe
#PBS -M bures024@umn.edu

cd ~/gameOfLife
module load python2
source activate pycuda3
module load cuda
python gpuGameOfLife.py