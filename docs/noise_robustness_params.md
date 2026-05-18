# 噪声鲁棒性实验参数方案

## 1. 实验目标

评估算法在不同噪声水平下的鲁棒性。基准条件（GT 图）模拟真实 STEM 实验的典型噪声水平（PSNR ≈ 20 dB），在此基础上逐步增加噪声强度，观察算法性能退化。

## 2. 噪声模型

### 2.1 三种噪声源

实际 STEM 成像中的噪声由三种来源组成：

| 噪声类型 | 物理来源 | 对应函数 |
|---------|---------|---------|
| 扫描噪声 | 电子束逐行扫描的位置抖动 | `add_scan_noise_from_original()` |
| 泊松噪声 | 电子计数的统计涨落 | `add_poisson_noise_from_original()` |
| 高斯噪声 | 探测器读出噪声及其他加性噪声 | `add_gaussian_noise()` |

### 2.2 扫描噪声与泊松噪声参数（固定，最小存在感）

实验中扫描噪声和泊松噪声参数固定不变，仅提供"类型存在感"以确保模型在训练时见过全部噪声模式。这两类噪声的贡献极小（组合 PSNR ≈ 48 dB），几乎不影响总体 PSNR。

```yaml
scan_noise:
  pixel_size_A_orig: 0.12
  dwell_time_scan_s_orig: 3.0e-6
  sigma_jitter_A: 0.02
  width_orig_px: 3289
  line_freq_hz: 60.0
  phase_x: 0.0
  phase_y: 1.5707963267948966        # π/2

poisson_noise:
  beam_current_A: 3.0e-11
  dwell_time_s: 3.0e-6
  k: 7.8125                          # 2 * crop_side / 128 = 2 * 500 / 128
```

### 2.3 高斯噪声参数（控制变量）

通过调整 `gaussian_sigma` 参数来精确控制总体噪声水平。使用 `preserve_scale=True` 模式（噪声幅度按图像的动态范围缩放）。

## 3. GT 图（基准）

$$
\text{GT} = \text{clean} + n_{\text{scan}} + n_{\text{poisson}} + n_{\text{gaussian}}(\sigma_1)
$$

其中 $\sigma_1 = 0.129$，对应平均 PSNR(GT vs clean) ≈ **20.0 dB**（std = 1.19 dB）。

### 3.1 σ₁ 的校准

在 `data/experiments/ReS2_noise0_defect0/` 全部 1000 张无噪声 ReS2 图像上（100 张采样精搜）统计：

- 图像动态范围 (vmax − vmin)：mean=0.918, std=0.110, range=[0.50, 0.98]
- `gaussian_sigma=0.129` 产生的平均 PSNR(vs clean) = 19.97 dB, std=1.17 dB

## 4. 鲁棒性实验档次

### 4.1 核心公式

在 GT 图上叠加额外高斯噪声 $n_2 \sim \mathcal{N}(0, \sigma_2^2)$：

$$
\text{noisy} = \text{GT} + n_2
$$

由高斯噪声的可加性，等价于直接在 clean 图上加 $\sigma_{\text{total}}$ 的噪声：

$$
\text{noisy} = \text{clean} + n_{\text{scan}} + n_{\text{poisson}} + n_{\text{gaussian}}(\sigma_{\text{total}})
$$

$$
\sigma_{\text{total}} = \sqrt{\sigma_1^2 + \sigma_2^2}
$$

### 4.2 实验参数查找表（6 个档次）

横轴为 **PSNR(noisy vs GT)**（整数），通过控制 $\sigma_2$（或等价地 $\sigma_{\text{total}}$）实现。

| PSNR(vs GT) | σ₂ | σ_total | PSNR(vs clean)¹ | 数据集目录 |
|:---:|:---:|:---:|:---:|:---|
| ∞ (GT) | 0 | 0.129 | 20.0 dB | `ReS2_noise_gt_defect0` |
| **30 dB** | 0.038 | 0.1345 | 19.7 dB | `ReS2_noise_30_defect0` |
| **26 dB** | 0.060 | 0.1423 | 19.3 dB | `ReS2_noise_26_defect0` |
| **22 dB** | 0.098 | 0.1620 | 18.2 dB | `ReS2_noise_22_defect0` |
| **18 dB** | 0.163 | 0.2079 | 16.3 dB | `ReS2_noise_18_defect0` |
| **15 dB** | 0.240 | 0.2725 | 14.2 dB | `ReS2_noise_15_defect0` |
| **12 dB** | 0.364 | 0.3862 | 11.7 dB | `ReS2_noise_12_defect0` |

¹ 在全部 1000 张图上的实测平均值。

### 4.3 完整 σ₂ 对照表（供扩展）

以下为 PSNR(vs GT) = 11–35 dB 全部整数档的校准值（校准于 100 张图采样）：

| PSNR(vs GT) | σ₂ | σ_total | PSNR(vs clean) |
|:---:|:---:|:---:|:---:|
| 35 | 0.021 | 0.1307 | 19.91 |
| 34 | 0.023 | 0.1310 | 19.89 |
| 33 | 0.026 | 0.1316 | 19.86 |
| 32 | 0.030 | 0.1324 | 19.81 |
| 31 | 0.033 | 0.1332 | 19.77 |
| 30 | 0.038 | 0.1345 | 19.69 |
| 29 | 0.042 | 0.1357 | 19.62 |
| 28 | 0.048 | 0.1376 | 19.50 |
| 27 | 0.054 | 0.1398 | 19.37 |
| 26 | 0.060 | 0.1423 | 19.22 |
| 25 | 0.068 | 0.1458 | 19.01 |
| 24 | 0.077 | 0.1502 | 18.76 |
| 23 | 0.087 | 0.1556 | 18.47 |
| 22 | 0.098 | 0.1620 | 18.13 |
| 21 | 0.111 | 0.1702 | 17.73 |
| 20 | 0.126 | 0.1803 | 17.25 |
| 19 | 0.143 | 0.1926 | 16.72 |
| 18 | 0.163 | 0.2079 | 16.11 |
| 17 | 0.185 | 0.2255 | 15.46 |
| 16 | 0.211 | 0.2473 | 14.74 |
| 15 | 0.240 | 0.2725 | 14.00 |
| 14 | 0.275 | 0.3038 | 13.19 |
| 13 | 0.316 | 0.3413 | 12.34 |
| 12 | 0.364 | 0.3862 | 11.48 |
| 11 | 0.425 | 0.4441 | 10.58 |

### 4.4 生成方式

**不需要** 分两步（先生成 GT 再叠加 σ₂），直接一步完成：

```python
noisy = add_scan_noise(clean, ...)       # 最小扫描噪声
noisy = add_poisson_noise(noisy, ...)    # 最小泊松噪声
noisy = add_gaussian_noise(noisy, sigma=sigma_total, preserve_scale=True)
```

生成脚本：`tools/generate_noise_robustness_datasets.py`

```bash
python tools/generate_noise_robustness_datasets.py \
  --input_dir data/experiments/ReS2_noise0_defect0 \
  --output_parent data/experiments \
  --levels 30,26,22,18,15,12 \
  --also-gt --seed 12345
```

## 5. 校准方法论

### 5.1 σ → PSNR 的非普适性

`gaussian_sigma` 与 PSNR 不是普适的一一对应关系，受以下因素影响：

- **图像动态范围**：`preserve_scale=True` 下，实际噪声幅度 = sigma × (vmax − vmin)
- **clip 操作**：`np.clip` 截断超出范围的噪声尾部，降低有效噪声能量
- **图像内容**：暗区与亮区的 clip 比例不同

### 5.2 校准方式

1. **σ₁ 校准**：在 1000 张实际实验图中采样 100 张，以 0.001 步长扫描 σ 值，选择使平均 PSNR(vs clean) 最接近 20 dB 的 σ。结果：σ₁ = 0.129。

2. **σ₂ 校准**：固定 σ₁ 生成 GT 图后，在 GT 上以 0.001 步长扫描 σ₂，测量 PSNR(noisy vs GT) 和 PSNR(noisy vs clean)。选择使 PSNR(vs GT) 为整数且 PSNR(vs clean) 在 [10, 20] 范围内的 σ₂。

3. **σ_total 计算**：由高斯可加性 $\sigma_{\text{total}} = \sqrt{\sigma_1^2 + \sigma_2^2}$。

### 5.3 PSNR(vs GT) 的计算

$$
\text{PSNR(vs GT)} = 10 \log_{10}\!\left(\frac{1}{\text{MSE}(\text{noisy}, \text{GT})}\right)
$$

其中 MSE 在 [0, 1] 归一化图像上计算（MAX_I = 1）。

> **注意**：在实际生成中，GT 和 noisy 使用不同的随机种子独立生成，因此逐张图计算的 PSNR(vs GT) 会因噪声 realization 差异而偏低。校准表中的 PSNR(vs GT) 是在"GT 上叠加 σ₂"的设定下测量的，反映的是额外噪声 σ₂ 的真实贡献。

## 6. 论文表述建议

> The baseline noise level (GT) simulates typical STEM experimental conditions with PSNR ≈ 20 dB relative to the noise-free simulated image. The full noise model includes scan distortion, Poisson counting noise, and Gaussian readout noise. To evaluate robustness, we progressively increase the Gaussian noise component while keeping scan and Poisson noise at minimal levels, measuring the additional degradation in PSNR relative to the GT image. Six noise levels are tested: PSNR(vs GT) = 30, 26, 22, 18, 15, and 12 dB, corresponding to PSNR(vs clean) ranging from 19.7 to 11.7 dB.
