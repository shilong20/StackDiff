# StackDiff

StackDiff is the official implementation of the paper **StackDiff: Human-like, physics-constrained unsupervised learning for picometer-accuracy layer-resolved stacking analysis**. It provides tools for STEM image layer separation, unconditional sampling, training configuration, synthetic STEM data generation, symbolic regression for extracting image superposition formulas, ReS2 stacking analysis examples, and Re vacancy defect detection in monolayer ReS2.

This repository includes configurations and example inputs for various materials studied in the paper:

```text
ReS2, MoS2, MoTe2, TaS2...
```

Pretrained checkpoints and example training data will be available via Zenodo.

## Overview

- `src/main.py`: Entry point for multilayer image separation based on YAML configuration files.
- `configs/separate/`: Image separation configurations for different materials.
- `configs/sample/`: Unconditional sampling configurations for different materials.
- `configs/train/`: Training configurations for different materials, using pregenerated monolayer/source STEM images with online augmentation during training.
- `data/examples/<material>/`: Small-scale demonstration inputs, with files named from `0.png` to `4.png`.
- `data/training_source/<material>/mask.png`: Mask required for training source image generation and online augmentation.
- `src/tools/synthetic_materials/`: Tools for synthetic STEM image generation, including structural files and generation configurations for different materials. These tools are used to generate simulated data for training diffusion models of monolayer van der Waals materials.
- `src/tools/res2_stacking_analysis/`: ReS2-specific examples for stacking, slip, and twist analysis.
- `src/tools/Symbolic-regression/`: Symbolic regression tools for extracting intensity superposition formulas between monolayer and multilayer images from simulated data.
- `src/tools/monolayer_defect_detection/`: Model training and inference tools for Re vacancy defect detection in monolayer ReS2.


## 安装

建议在独立 Python 环境中安装依赖：

```bash
pip install -r requirements.txt
```

训练和采样需要可用的 PyTorch/CUDA GPU 环境。

## 模型权重

每种材料默认使用一个推荐 EMA checkpoint。下载权重后，请按以下路径放置：

```text
models/checkpoints/ReS2/ema_0.9999_200000.pt
models/checkpoints/MoS2/ema_0.9999_200000.pt
models/checkpoints/MoTe2/ema_0.9999_200000.pt
models/checkpoints/TaS2/ema_0.9999_200000.pt
...
```

请将下载后的 checkpoint 放到上述路径；默认配置文件已经指向这些位置。

权重下载链接配置完成后，也可以使用脚本自动下载：

```bash
python scripts/download_weights.py --material all
```

## 图像分离

以下为默认示例输入。运行以下命令即可对 `data/examples/<material>/` 中的图像进行分离：

```bash
python src/main.py --config configs/separate/ReS2.yml
python src/main.py --config configs/separate/MoS2.yml
python src/main.py --config configs/separate/MoTe2.yml
python src/main.py --config configs/separate/TaS2.yml
...
```

默认输出目录为：

```text
outputs/separation/<material>/
```

每个输入图像对应一个输出文件夹，通常包含：

- `_original.png`：预处理后的输入图像。
- `_0.png`：分离得到的第 0 层。
- `_1.png`：分离得到的第 1 层。
- `_combine.png`：两层重叠重构图。

使用自定义输入时，可修改对应 YAML 中的：

```yaml
paths:
  default_input: ...
  default_output: ...
```

启用 auto-crop 时，若文件名包含物理视野后缀，例如 `sample-7.1x3.1.png`，代表该图像的物理尺寸为7.1 nm × 3.1 nm，程序会据此推断裁剪和 resize 设置。

## 无条件采样

下载对应材料的 checkpoint 后，可运行无条件采样：

```bash
python src/core/scripts/image_sample.py --config configs/sample/ReS2.yml
python src/core/scripts/image_sample.py --config configs/sample/MoS2.yml
python src/core/scripts/image_sample.py --config configs/sample/MoTe2.yml
python src/core/scripts/image_sample.py --config configs/sample/TaS2.yml
...
```

采样输出目录由 `configs/sample/<material>.yml` 中的 `sample.output_dir` 指定。

## 训练

训练使用预先生成的单层/source STEM 仿真图像，并在训练过程中动态施加随机增强。训练 loader 会从 source 图像中采样，经过 crop、rotation、elastic/perspective 形变、scan noise、display 变换等增强，得到 128 x 128 的训练 patch。

默认 source 数据目录为：

```text
data/training_source/<material>/
```

仓库已包含 `mask.png`。source 图像可使用下方 synthetic STEM 数据生成工具生成，也可替换为用户自己的仿真结果。

训练命令如下：

```bash
python src/core/scripts/image_train.py --config configs/train/ReS2.yml
python src/core/scripts/image_train.py --config configs/train/MoS2.yml
python src/core/scripts/image_train.py --config configs/train/MoTe2.yml
python src/core/scripts/image_train.py --config configs/train/TaS2.yml
```

训练 checkpoint 和日志会写入 `train.output_dir` 指定的目录。

## Synthetic STEM 数据生成

`src/tools/synthetic_materials/` 提供可选 synthetic STEM 生成工具，包括四种材料的 JSON、XYZ 和 mask 资产。

该生成工具依赖 computem/temsim 项目提供的外部可执行程序 `incostem`：

```text
https://sourceforge.net/projects/computem/files/
```

`incostem` 不随本仓库分发。可将本地可执行文件放在：

```text
src/tools/synthetic_materials/incostem
```

也可以在材料 JSON 中将 `incostem_path` 设置为绝对路径。

生成训练前使用的单层/source STEM 图像：

```bash
python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/source_configs/ReS2.json
python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/source_configs/MoS2.json
python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/source_configs/MoTe2.json
python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/source_configs/TaS2.json
```

source generator 会在 `data/training_source/<material>/` 下写入 PNG、`mask.png` 和 `manifest.jsonl`。

生成用于图像分离示例或评估的 synthetic 双层 PNG：

```bash
python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/ReS2.json
python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/MoS2.json
python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/MoTe2.json
python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/TaS2.json
```

## ReS2 堆垛解析示例

`src/tools/res2_stacking_analysis/` 提供 ReS2 专用的堆垛解析示例，包括：

- 原子位置识别。
- 单层晶格矢量估计。
- slip 样本的 da/db 投影。
- twist、slip、flip-twist 和 flip-slip 分类。
- 基于解析结果的可视化。

该堆垛解析工具目前仅适用于 ReS2。对于其他材料，本仓库提供分离、采样、训练和 synthetic 数据生成配置。

解析工具要求每个分离后的双层样本放在独立文件夹中，并包含一对分离层图像：

```text
outputs/separation/ReS2/0/
  0_original.png
  0_0.png
  0_1.png
  0_combine.png
```

其中 `*_0.png` 和 `*_1.png` 是必需文件。

样本文件夹名可以包含物理视野信息，格式为 `<width_nm>x<height_nm>`，例如：

```text
0_2.79x2.79/
```

存在这类 tag 时，工具会将 nm 单位的视野尺寸转换为 Angstrom，例如由 `2.79 nm` 推断 `sideA_A = 27.9`。如果没有尺寸 tag，默认使用：

```bash
--sideA_A 27.9
```

如果图像对应不同物理视野，应手动覆盖该参数。

运行堆垛分类和单层晶格提取：

```bash
python src/tools/res2_stacking_analysis/classify_bilayers.py \
  --root outputs/separation/ReS2 \
  --out_csv outputs/analysis/ReS2/res2_stacking.csv \
  --atoms_out_dir outputs/analysis/ReS2/atoms
```

计算层间几何量：

```bash
python src/tools/res2_stacking_analysis/analyze_interlayer.py \
  --root outputs/separation/ReS2 \
  --out_csv outputs/analysis/ReS2/interlayer.csv \
  --stacking_csv outputs/analysis/ReS2/res2_stacking.csv \
  --stacking_atoms_dir outputs/analysis/ReS2/atoms
```

对于 slip 类样本，脚本会输出像素位移、物理位移，以及在 `(da, db)` 基底下的投影。对于 twist 类样本，脚本会输出 refined twist angle。

也可以只运行 `analyze_interlayer.py`，并让它在需要时自动调用堆垛分类：

```bash
python src/tools/res2_stacking_analysis/analyze_interlayer.py \
  --root outputs/separation/ReS2 \
  --out_csv outputs/analysis/ReS2/interlayer.csv \
  --force_stacking
```

生成解析可视化，包括原子点、单层晶格矢量，以及 slip/flip-slip 对齐结果：

```bash
python src/tools/res2_stacking_analysis/visualize_debug.py \
  --root outputs/separation/ReS2 \
  --stacking_csv outputs/analysis/ReS2/res2_stacking.csv \
  --atoms_dir outputs/analysis/ReS2/atoms \
  --interlayer_csv outputs/analysis/ReS2/interlayer.csv \
  --out_dir outputs/analysis/ReS2/visualization
```

## 许可

本项目代码使用 MIT License 发布。详见 `LICENSE`。

数据集和模型权重的许可信息将随 Zenodo 记录一并说明。

## 致谢

StackDiff 借鉴了 diffusion inverse-problem 方法和 OpenAI guided-diffusion 的相关思想与代码组织方式。computem/temsim 等外部仿真工具是独立项目，不随本仓库分发。
