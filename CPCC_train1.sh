#!/bin/sh

python main.py --device cuda:0 --dataset CPCC0 --vid_res 1280x720 --mask_root /home/galinezh/TCVAE_modified/data/CPCC0/gt/test_frame_mask --seg_len 30 --seg_stride 15 --model_save_dir /home/galinezh/TCVAE_modified/VAE_results/CPCC0 --tcvae --mss --graph --model_lr 5e-5
python main.py --device cuda:0 --dataset CPCC1 --vid_res 1280x720 --mask_root /home/galinezh/TCVAE_modified/data/CPCC1/gt/test_frame_mask --seg_len 30 --seg_stride 15 --model_save_dir /home/galinezh/TCVAE_modified/VAE_results/CPCC1 --tcvae --mss --graph --model_lr 5e-5
python main.py --device cuda:0 --dataset CPCC2 --vid_res 1280x720 --mask_root /home/galinezh/TCVAE_modified/data/CPCC2/gt/test_frame_mask --seg_len 30 --seg_stride 15 --model_save_dir /home/galinezh/TCVAE_modified/VAE_results/CPCC2 --tcvae --mss --graph --model_lr 5e-5




