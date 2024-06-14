#!/bin/bash
#SBATCH --job-name=primitive_NLP_dataset_L=5
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mgallo@sissa.it
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=32G
#SBATCH --partition=regular1,regular2
#SBATCH --output=./log_opt/%x.o%A-%a
#SBATCH --error=./log_opt/%x.o%A-%a

python -u create_dataset.py