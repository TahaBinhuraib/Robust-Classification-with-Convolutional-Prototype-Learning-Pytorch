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

python main.py --batch_size 64 --h 2 --lr 0.0001 --exp_name h_2_lr_1e4 &
sleep 30 &
python main.py --batch_size 64 --h 4 --lr 0.0001 --exp_name h_4_lr_1e4 &
sleep 30 &
python main.py --batch_size 64 --h 8 --lr 0.0001 --exp_name h_8_lr_1e4 &
sleep 30 &
python main.py --batch_size 64 --h 16 --lr 0.0001 --exp_name h_16_lr_1e4 &
sleep 30 &
python main.py --batch_size 64 --h 32 --lr 0.0001 --exp_name h_32_lr_1e4 &
sleep 30 &
python main.py --batch_size 64 --h 2 --lr 0.00001 --exp_name h_2_lr_1e5 &
sleep 30 &
python main.py --batch_size 64 --h 4 --lr 0.00001 --exp_name h_4_lr_1e5 &
sleep 30 &
python main.py --batch_size 64 --h 6 --lr 0.00001 --exp_name h_8_lr_1e5 &
sleep 30 &
python main.py --batch_size 64 --h 16 --lr 0.00001 --exp_name h_16_lr_1e5 &
sleep 30 &
python main.py --batch_size 64 --h 32 --lr 0.00001 --exp_name h_32_lr_1e5 &
sleep 30 &
python main.py --batch_size 64 --h 64 --lr 0.0001 --exp_name h_64_lr_1e4 &
sleep 30 &
python main.py --batch_size 64 --h 128 --lr 0.0001 --exp_name h_128_lr_1e4 &
sleep 30 &
python main.py --batch_size 64 --h 256 --lr 0.0001 --exp_name h_256_lr_1e4 &
sleep 30 &
python main.py --batch_size 64 --h 64 --lr 0.00001 --exp_name h64_lr_1e5 &
sleep 30 &
python main.py --batch_size 64 --h 128 --lr 0.00001 --exp_name h_128_lr_1e5 &
sleep 30 &
python main.py --batch_size 64 --h 256 --lr 0.00001 --exp_name h_256_lr_1e5 
