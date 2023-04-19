#!/bin/bash

#SBATCH -A yzmkp1
#SBATCH -p a100q
#SBATCH -o txt_output/%j_out.txt
#SBATCH -e txt_output/%j_err.txt
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1         # Gerekli GPU sayısı (GPU kullanmayacaksanız 0 olarak değeri giriniz.)

module load cuda/cuda-11.7-a100q
module load ANACONDA/Anaconda3-2022.10-python-3.9

source activate diffusion-prototype

python main.py --batch_size 128 --h 16 --lr 0.0001 --reg 0.01 --exp_name h_16_lr_1e4_reg_1e2 &
sleep 30 &
python main.py --batch_size 128 --h 32 --lr 0.0001 --reg 0.01 --exp_name h_32_lr_1e4_reg_1e2 &
sleep 30 &
python main.py --batch_size 128 --h 64 --lr 0.0001 --reg 0.01 --exp_name h_64_lr_1e4_reg_1e2 &
sleep 30 &
python main.py --batch_size 128 --h 128 --lr 0.0001 --reg 0.01 --exp_name h_128_lr_1e4_reg_1e2 &
sleep 30 &
python main.py --batch_size 128 --h 256 --lr 0.0001 --reg 0.01 --exp_name h_256_lr_1e4_reg_1e2 

