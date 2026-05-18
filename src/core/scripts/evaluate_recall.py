import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import os
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import argparse

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用预训练的ResNet作为特征提取器
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # 移除最后的全连接层
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
    def forward(self, x):
        return self.model(x).squeeze()

def load_and_extract_features(image_dir, feature_extractor, device):
    """
    加载图像并提取特征
    """
    features = []
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    print(f"Found {len(image_files)} images in {image_dir}")
    
    feature_extractor.eval()
    with torch.no_grad():
        for img_path in tqdm(image_files):
            # 读取灰度图
            img = Image.open(os.path.join(image_dir, img_path))
            img = np.array(img)
            
            # 转换为3通道（通过复制通道）
            img = np.stack([img] * 3, axis=-1)
            
            # 预处理
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            img = img.unsqueeze(0).to(device)
            
            # 提取特征
            feat = feature_extractor(img)
            features.append(feat.cpu().numpy())
            
            # 定期清理GPU内存
            if len(features) % 100 == 0:
                torch.cuda.empty_cache()
    
    features = np.stack(features)
    print(f"Extracted features shape: {features.shape}")
    return features

def calculate_recall(real_features, generated_features, k=5):
    """
    计算recall
    """
    print("Calculating recall...")
    print(f"Real features shape: {real_features.shape}")
    print(f"Generated features shape: {generated_features.shape}")
    
    # 归一化特征向量
    real_features = real_features / np.linalg.norm(real_features, axis=1, keepdims=True)
    generated_features = generated_features / np.linalg.norm(generated_features, axis=1, keepdims=True)
    
    # 使用KNN找到最近邻
    nbrs = NearestNeighbors(n_neighbors=k, metric='cosine')
    nbrs.fit(generated_features)
    
    distances, _ = nbrs.kneighbors(real_features)
    
    # 打印距离统计信息
    print(f"Distance stats: mean={distances.mean():.4f}, min={distances.min():.4f}, max={distances.max():.4f}")
    
    # 计算recall
    threshold = 0.01  # 使用更严格的阈值
    recalled = (distances < threshold).any(axis=1)
    recall = recalled.mean()
    
    return recall

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", type=str, required=True, help="Path to real images")
    parser.add_argument("--generated_path", type=str, required=True, help="Path to generated images")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for feature extraction")
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化特征提取器
    feature_extractor = FeatureExtractor().to(device)
    
    # 加载图像并提取特征
    real_features = load_and_extract_features(args.real_path, feature_extractor, device)
    generated_features = load_and_extract_features(args.generated_path, feature_extractor, device)
    
    print(f"Loaded {real_features.shape[0]} real images and {generated_features.shape[0]} generated images")
    print(f"Feature dimension: {real_features.shape[1]}")
    
    # 计算不同k值下的recall
    k_values = [1, 5, 10, 20]
    for k in k_values:
        recall = calculate_recall(real_features, generated_features, k=k)
        print(f"Recall@{k}: {recall:.4f}")
        
        # 保存结果
        with open("evaluation_recall_results.txt", "a") as f:
            f.write(f"Recall@{k}: {recall:.4f}\n")

if __name__ == "__main__":
    main() 