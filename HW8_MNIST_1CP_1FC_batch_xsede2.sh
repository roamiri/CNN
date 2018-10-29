#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --qos=normal
#SBATCH --ntasks-per-node 7
#SBATCH --gres=gpu:k80:1
#SBATCH -t 1:00:00
# Join output and errors into output.
#SBATCH -o chiasson.o%j
#SBATCH -e chiasson.e%j
# echo commands to stdout (standard output)
set -x
# move to the working directory
# cd $SCRATCH
# copy the code and data to the working directory
# cp -R $SCRATCH/Michael_A_Nielsen
# go to the directory where actual runfile is present
cd $HOME/Michael_A_Nielsen/MichaelNielsen_code_chap6/
# load all necessary modules
module load python cuda theano
# run the python code and copy output file to persistent storage
python Run_Michael_Nielsen_MNIST_1CP_1FC_Softmax_program.py






