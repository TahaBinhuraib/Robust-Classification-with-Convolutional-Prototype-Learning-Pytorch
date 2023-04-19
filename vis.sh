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

python visualize.py --checkpoint results/resnet18/h_2_lr_1e4_reg_1e3/best_model.pt --save results/resnet18/h_2_lr_1e4_reg_1e3 --hidden_units 2 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_4_lr_1e4_reg_1e3/best_model.pt --save results/resnet18/h_4_lr_1e4_reg_1e3 --hidden_units 4 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_8_lr_1e4_reg_1e3/best_model.pt --save results/resnet18/h_8_lr_1e4_reg_1e3 --hidden_units 8 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_16_lr_1e4_reg_1e3/best_model.pt --save results/resnet18/h_16_lr_1e4_reg_1e3 --hidden_units 16 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_32_lr_1e4_reg_1e3/best_model.pt --save results/resnet18/h_32_lr_1e4_reg_1e3 --hidden_units 32 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_64_lr_1e4_reg_1e3/best_model.pt --save results/resnet18/h_64_lr_1e4_reg_1e3 --hidden_units 64 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_128_lr_1e4_reg_1e3/best_model.pt --save results/resnet18/h_128_lr_1e4_reg_1e3 --hidden_units 128 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_256_lr_1e4_reg_1e3/best_model.pt --save results/resnet18/h_256_lr_1e4_reg_1e3 --hidden_units 256 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_2_lr_1e5_reg_1e3/best_model.pt --save results/resnet18/h_2_lr_1e5_reg_1e3 --hidden_units 2 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_4_lr_1e5_reg_1e3/best_model.pt --save results/resnet18/h_4_lr_1e5_reg_1e3 --hidden_units 4 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_8_lr_1e5_reg_1e3/best_model.pt --save results/resnet18/h_8_lr_1e5_reg_1e3 --hidden_units 8 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_16_lr_1e5_reg_1e3/best_model.pt --save results/resnet18/h_16_lr_1e5_reg_1e3 --hidden_units 16 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h _32_lr_1e5_reg_1e3/best_model.pt --save results/resnet18/h_32_lr_1e5_reg_1e3 --hidden_units 32 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_64_lr_1e5_reg_1e3/best_model.pt --save results/resnet18/h_64_lr_1e5_reg_1e3 --hidden_units 64 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_128_lr_1e5_reg_1e3/best_model.pt --save results/resnet18/h_128_lr_1e5_reg_1e3 --hidden_units 128 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_256_lr_1e5_reg_1e3/best_model.pt --save results/resnet18/h_256_lr_1e5_reg_1e3 --hidden_units 256 
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_16_lr_1e4_reg_1e2/best_model.pt --save results/resnet18/h_16_lr_1e4_reg_1e2 --hidden_units 16 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_32_lr_1e4_reg_1e2/best_model.pt --save results/resnet18/h_32_lr_1e4_reg_1e2 --hidden_units 32 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_64_lr_1e4_reg_1e2/best_model.pt --save results/resnet18/h_64_lr_1e4_reg_1e2 --hidden_units 64 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_128_lr_1e4_reg_1e2/best_model.pt --save results/resnet18/h_128_lr_1e4_reg_1e2 --hidden_units 128 &
sleep 30 &
python visualize.py --checkpoint results/resnet18/h_256_lr_1e4_reg_1e2/best_model.pt --save results/resnet18/h_256_lr_1e4_reg_1e2 --hidden_units 256
