"""
Evaluate the diffusion model using FID score
"""

import argparse
import os
import sys
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

# 将当前脚本所在目录添加到Python模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def calculate_fid(real_image_dir, generated_image_dir, device='cuda'):
    """
    计算FID分数
    Args:
        real_image_dir: 真实图像目录
        generated_image_dir: 生成图像目录
        device: 计算设备
    Returns:
        fid_score: FID分数
    """
    print(f"Calculating FID between {real_image_dir} and {generated_image_dir}")
    
    # 初始化FID
    fid = FrechetInceptionDistance(normalize=True).to(device)
    
    # 加载真实图像
    print("Loading real images...")
    real_images = []
    for img_path in tqdm(os.listdir(real_image_dir)):
        if not img_path.endswith(('.png', '.jpg', '.jpeg')):
            continue
        # 读取灰度图
        img = Image.open(os.path.join(real_image_dir, img_path))
        img = np.array(img)
        # 将单通道转换为三通道
        img = np.stack([img] * 3, axis=-1)
        real_images.append(torch.tensor(img).permute(2, 0, 1))
        if len(real_images) % 100 == 0:
            # 批量处理以节省内存
            batch = torch.stack(real_images).to(device)
            fid.update(batch, real=True)
            real_images = []
    
    if real_images:  # 处理剩余的图像
        batch = torch.stack(real_images).to(device)
        fid.update(batch, real=True)
    
    # 加载生成的图像
    print("Loading generated images...")
    fake_images = []
    for img_path in tqdm(os.listdir(generated_image_dir)):
        if not img_path.endswith(('.png', '.jpg', '.jpeg')):
            continue
        # 读取灰度图
        img = Image.open(os.path.join(generated_image_dir, img_path))
        img = np.array(img)
        # 将单通道转换为三通道
        img = np.stack([img] * 3, axis=-1)
        fake_images.append(torch.tensor(img).permute(2, 0, 1))
        if len(fake_images) % 100 == 0:
            # 批量处理以节省内存
            batch = torch.stack(fake_images).to(device)
            fid.update(batch, real=False)
            fake_images = []
    
    if fake_images:  # 处理剩余的图像
        batch = torch.stack(fake_images).to(device)
        fid.update(batch, real=False)
    
    # 计算FID
    fid_score = float(fid.compute())
    print(f"FID Score: {fid_score}")
    return fid_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", type=str, required=True, help="Path to real images")
    parser.add_argument("--generated_path", type=str, required=True, help="Path to generated images")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    # 计算FID
    fid_score = calculate_fid(args.real_path, args.generated_path, args.device)
    
    # 保存结果
    with open("evaluation_FID_results.txt", "a") as f:
        f.write(f"FID Score between {args.real_path} and {args.generated_path}: {fid_score}\n")

if __name__ == "__main__":
    main()