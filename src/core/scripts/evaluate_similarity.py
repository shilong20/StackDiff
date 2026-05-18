import numpy as np
from scipy.stats import wasserstein_distance
import random
from tqdm import tqdm
import argparse
import os
from PIL import Image

def evaluate_datasets_similarity(real_images, generated_images, n_samples=1000):
    """
    评估两个图像数据集之间的相似性，仅计算Wasserstein距离
    
    Parameters:
    - real_images: 真实图像列表
    - generated_images: 生成图像列表
    - n_samples: 随机采样对数量
    """
    metrics = {}
    
    # 计算整体分布的Wasserstein距离
    print("Calculating Wasserstein distance...")
    
    # 计算图像整体灰度分布的Wasserstein距离
    real_pixels = np.concatenate([img.flatten() for img in real_images])
    gen_pixels = np.concatenate([img.flatten() for img in generated_images])
    
    w_distance = wasserstein_distance(real_pixels, gen_pixels)
    metrics['Wasserstein_global'] = w_distance
    
    return metrics

def main():
    print("Starting evaluation...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", type=str, required=True)
    parser.add_argument("--generated_path", type=str, required=True)
    args = parser.parse_args()
    
    print(f"Arguments parsed: real_path={args.real_path}, generated_path={args.generated_path}")
    
    # 加载图像
    print("Loading real images...")
    real_images = load_images(args.real_path)
    print("Loading generated images...")
    generated_images = load_images(args.generated_path)
    
    print(f"Loaded {len(real_images)} real images and {len(generated_images)} generated images")
    
    # 计算评估指标
    metrics = evaluate_datasets_similarity(real_images, generated_images)
    
    # 打印结果
    print("\nDataset Similarity Metrics:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric:30s}: {value:.4f}")
    
    # 保存结果
    with open("dataset_similarity_results.txt", "a") as f:
        f.write("Dataset Similarity Metrics:\n")
        f.write("-" * 50 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric:30s}: {value:.4f}\n")


def load_images(path):
    """加载图像并进行预处理"""
    images = []
    print(f"Loading images from {path}")
    for img_path in tqdm(os.listdir(path)):
        if img_path.endswith('.png'):
            img = Image.open(os.path.join(path, img_path))
            img = np.array(img) / 255.0  # 归一化到[0,1]
            images.append(img)
    return images

if __name__ == "__main__":
    main() 