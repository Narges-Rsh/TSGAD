#!/bin/sh

python main.py --device cuda:1 --dataset CPCC3 --vid_res 1280x720 --mask_root /home/galinezh/TCVAE_modified/data/CPCC3/gt/test_frame_mask --seg_len 30 --seg_stride 15 --model_save_dir /home/galinezh/TCVAE_modified/VAE_results/CPCC3 --tcvae --mss --graph --model_lr 5e-5
python main.py --device cuda:1 --dataset CPCC4 --vid_res 1280x720 --mask_root /home/galinezh/TCVAE_modified/data/CPCC4/gt/test_frame_mask --seg_len 30 --seg_stride 15 --model_save_dir /home/galinezh/TCVAE_modified/VAE_results/CPCC4 --tcvae --mss --graph --model_lr 5e-5
python main.py --device cuda:1 --dataset CPCC5 --vid_res 1280x720 --mask_root /home/galinezh/TCVAE_modified/data/CPCC5/gt/test_frame_mask --seg_len 30 --seg_stride 15 --model_save_dir /home/galinezh/TCVAE_modified/VAE_results/CPCC5 --tcvae --mss --graph --model_lr 5e-5
python main.py --device cuda:1 --dataset CPCC6 --vid_res 1280x720 --mask_root /home/galinezh/TCVAE_modified/data/CPCC6/gt/test_frame_mask --seg_len 30 --seg_stride 15 --model_save_dir /home/galinezh/TCVAE_modified/VAE_results/CPCC6 --tcvae --mss --graph --model_lr 5e-5









