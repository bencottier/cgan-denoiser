#!/bin/bash
#SBATCH --job-name=cgan_train
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bcottier2@gmail.com
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

conda activate tf-biomed
cd ~/honours/nn-artefact-removal
python test_undersampled.py

