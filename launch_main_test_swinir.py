import os
import sys

import subprocess
import argparse
from torchvision.datasets import ImageFolder
import numpy as np
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch main test swinir')
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--nprocs', type=int, default=1, help='Number of processes to use for testing')
    parser.add_argument('--save_dir', type=str, default=None, help='Output directory for saving results')
    args = parser.parse_args()
    
    
    folder = ImageFolder(args.folder_lq)
    num_idxs_per_proc = int(np.ceil(len(folder)/float(args.nprocs)))
    assert args.nprocs <= torch.cuda.device_count(), 'Number of processes is larger than the number of GPUs'
    
    for i in range(args.nprocs):
        proc_arr = ['python', 'main_test_swinir.py', '--task', 'real_sr', '--scale', str(args.scale), '--model_path', args.model_path, '--idx_range', str(i*num_idxs_per_proc), str((i+1)*num_idxs_per_proc), '--save_dir', args.save_dir, '--device_idx', str(i)]
        if args.folder_lq is not None:
            proc_arr += ['--folder_lq', args.folder_lq]
        if args.folder_gt is not None:
            proc_arr += ['--folder_gt', args.folder_gt]
        if args.tile is not None:
            proc_arr += ['--tile', str(args.tile)]
        if i == 0:
            proc_arr += ['--alert']