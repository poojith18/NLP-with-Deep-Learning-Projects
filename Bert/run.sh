#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --mail-user=u1405749@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_4-%j
#SBATCH --export=ALL
source ~/miniconda3/etc/profile.d/conda.sh
conda activate DLNLP1
OUT_DIR=/scratch/general/vast/u1405749/cs6957/assignment4/models
mkdir -p ${OUT_DIR}
python main_rte_WFT.py --output_dir ${OUT_DIR}
python main_rte_WOFT.py --output_dir ${OUT_DIR}
python main_sst2_WFT.py --output_dir ${OUT_DIR}
python main_sst2_WOFT.py --output_dir ${OUT_DIR}