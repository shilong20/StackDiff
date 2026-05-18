# Methods: Model Training

## Training data: simulation-to-experiment augmentation pipeline

StackDiff 采用纯仿真数据训练策略：无需任何实验标注数据，仅通过仿真与在线增强即可学习忠实的单层 STEM 图像先验，并泛化到实验图像。

训练数据由两阶段管线在线生成。首先，使用 incostem 多切片仿真程序从晶体结构文件生成高分辨率单层 HAADF-STEM 仿真图像，并引入随机空位与冻结声子近似下的原子位置扰动以反映真实晶格无序性。随后，在训练过程中对每张仿真图像实时施加在线数据增强管线，依次包含弹性形变、透视变换、非晶碳背景、几何变换（旋转/裁剪/翻转）、物理噪声链与显示级调整，弥合仿真与实验图像之间的域差距（管线细节与消融实验见 Supplementary Note 2、Supplementary Figs. 2–3）。

## Model architecture

扩散模型的去噪网络采用 U-Net 架构，输入输出均为单通道灰度 128×128 图像。网络设计遵循 Dhariwal & Nichol 的架构：基础通道数 256，五个分辨率级别的通道倍率为 (1, 1, 2, 3, 4)，对应最大通道数 1024；每个分辨率级别包含 2 个残差块；在 32×32、16×16 和 8×8 三个特征图分辨率上使用 4 头自注意力机制。网络采用 FiLM 自适应归一化（scale-shift norm）将时间步信息注入特征图，使用基于残差块的上下采样（resblock up/downsampling），并预测噪声与方差两个通道（learned sigma）。模型总参数量约 420M。

## Training objective

训练采用标准的去噪扩散概率模型（DDPM）框架。前向过程按照线性方差调度在 T = 1000 个时间步内向干净图像逐步添加高斯噪声。训练目标为简化的噪声预测损失：在每个随机采样的时间步 t，U-Net 学习从含噪图像 x_t 中预测所添加的噪声 ε，最小化均方误差 E[||ε − ε_θ(x_t, t)||²]。时间步采样策略为均匀采样（uniform sampler）。由于模型同时预测方差，实际训练损失还包含与可学习方差相关的变分下界项。

## Training setup

模型使用 AdamW 优化器（学习率 1×10⁻⁴，权重衰减为 0）训练 200,000 步，batch size 为 32。采用混合精度训练（FP16）以提升训练效率和降低显存占用。指数移动平均（EMA，衰减系数 0.9999）用于推理时的模型权重平滑。训练在单块 NVIDIA GPU 上完成。在线数据增强管线使用 64 个 CPU worker 进程并行生成训练样本，确保 GPU 利用率不受数据加载瓶颈限制。模型每 10,000 步保存一次检查点。

推理时采用 200 步 DDIM（Denoising Diffusion Implicit Models）加速采样。