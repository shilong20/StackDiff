# StackDiff: 多层材料分解与分析

> 基于扩散模型和物理约束的二维材料层分解系统。论文实验覆盖 ReS2、MoS2、MoTe2、TaS2 四种材料。

本仓库是论文开源版代码工作区：只包含核心代码、公开配置、少量示例数据和结构文件。完整训练数据、完整实验结果和模型权重不放入 Git；每种材料只发布一个推荐 EMA 权重，下载说明见 `docs/download_weights.md`。

## 快速开始

```bash
pip install -r requirements.txt
```

下载权重后，将它们放到以下路径：

```text
models/checkpoints/ReS2/ema_0.9999_200000.pt
models/checkpoints/MoS2/ema_0.9999_200000.pt
models/checkpoints/MoTe2/ema_0.9999_200000.pt
models/checkpoints/TaS2/ema_0.9999_200000.pt
```

运行 ReS2 示例分解：

```bash
python src/main.py --config configs/separate/ReS2.yml
```

输出默认写入：

```text
outputs/separation/ReS2/
```

生成少量模型采样：

```bash
python src/core/scripts/image_sample.py --config configs/sample/ReS2.yml
```

四种论文材料均提供采样配置：`ReS2.yml`、`MoS2.yml`、`MoTe2.yml`、`TaS2.yml`。

STEM 仿真样本生成：

```bash
python generate_sample/Batch_generate.py --config generate_sample/ReS2.json
```

`generate_sample` 需要外部 `incostem` 可执行文件；本仓库不分发该二进制和第三方源码，详见 `docs/data_examples.md`。

## ✨ 项目特性

- 🔧 **YAML 驱动**：所有参数集中在配置文件，最小化命令行
- 🧪 **论文材料**：`ReS2`、`MoS2`、`MoTe2`、`TaS2`
- ⚡ **仿真生成训练样本**：CPU 数据生成 + GPU 训练，高效利用资源
- 🎯 **物理约束**：DDNM 算子确保输出物理合理性
- 🔄 **Time-Travel Back**：RePaint 风格增强，提升约束条件下稳定性

### 核心使用流程

#### 1️⃣ 分解多层图像（最常用）

```bash
# 准备输入：复制图片到 data/multilayer/ReS2/
# 运行分解
python src/main.py --config configs/separate/ReS2_MoS2.yml

# 查看结果：data/results/ReS2/<图片名>/
#   ├── _original.png  # 输入原图
#   ├── _0.png         # 第0层
#   ├── _1.png         # 第1层
#   └── _combine.png   # 合并结果
```

#### 2️⃣ 训练模型

```bash
# 在线训练（边训练边生成数据）
python src/core/scripts/image_train.py --config configs/train/ReS2.yml
```

#### 3️⃣ 模型采样

```bash
# 生成样本到 ./sample/ 目录
python src/core/scripts/image_sample.py --config configs/sample/TaS2.yml
```

## 📚 完整命令参考

### 训练进阶

```bash
# 指定 GPU 训练
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python \
  src/core/scripts/image_train.py --config configs/train/ReS2.yml

# 在训练 YAML 的 train 段设置 gpu: 0（等效效果）
```

### 采样进阶

```bash
# 指定 GPU
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  python src/core/scripts/image_sample.py --config configs/sample/ReS2.yml

# 兼容旧参数法
python src/core/scripts/image_sample.py \
  --model_path models/checkpoints/xxx.pt \
  --num_samples 10 --image_size 128 --batch_size 4
```

### 分解进阶

```bash
# 指定 GPU
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  python src/main.py --config configs/separate/ReS2_test.yml

# 仅处理指定文件（在配置的 processing.selected_files 中列出）
```

### 模型评估

```bash
# FID 评估
python src/core/scripts/evaluate_FID.py --help

# 训练日志分析
python tools/analyze_training_loss.py --log_file models/checkpoints/ReS2/progress.csv

# 仅数值分析，不生成图表
python tools/analyze_training_loss.py --no-plot
```

### 图像处理工具

```bash
# 调整图像尺寸（保持长宽比）
python tools/resize_short_edge.py \
  --input_dir data/multilayer/ReS2 \
  --output_dir data/multilayer/ReS2 \
  --size 512
```

## 📖 核心概念

### Time-Travel Back
采样过程中的周期性回溯机制，显著提升约束条件下（如分解）的稳定性：

```yaml
sampling:
  time_travel:
    enable: true          # 开启/关闭
    travel_length: 10     # 回跳步长（建议 6-20）
    travel_repeat: 2      # 重复次数（>1 才生效）
```

**工作机制**：在总步数 `T_sampling` 内，以 `travel_length` 为间隔插入回跳并重新加噪（RePaint 风格），随后再继续前向采样。适用于分解、修复等带退化算子的任务。

### 自动裁剪
根据文件名中的物理尺寸（如 `image-7.1x3.1.png`，单位是nm，表示宽为7.1nm，高为3.1nm）自动计算最优裁剪区域，确保不同分辨率图像的处理一致性。

```yaml
processing:
  auto_crop:
    enabled: true
    unit_size_range: [2.4, 4.8]  # 单位尺寸范围（nm）
```

### 滑窗推理
通过重叠窗口处理高分辨率图像，在保证细节的同时避免显存溢出。

```yaml
processing:
  sliding_window: true  # 设为 false 使用整块推理（更快但可能有边缘效应）
```

### 自适应叠加算子（Adaptive Superposition）
- 适用于 `separate_sum` 任务，确保滑窗/整图在每一步迭代都满足 `max(k * ΣL_i) = max(y)` 的物理约束
- 在 YAML 的 `separation` 段开启：

```yaml
separation:
  method: separate_sum
  adaptive_superposition: true   # 默认 false，保持向后兼容
  # 若需要更稳定的求解，可改为手动指定常数 k（覆盖自适应）
  # 取值自动截断到 [0,1]，推荐与 adaptive_superposition 二选一
  fixed_superposition_k: 0.85    # 不填则使用自适应/默认 1.0
```

- 开启自适应时每次迭代都会重新计算 `k`；设置 `fixed_superposition_k` 时则全程复用同一常数。`*_conbine.png` 也使用相同算子，保证与输入观测一致的亮度/对比度。
- 固定 k 示例（以 `configs/separate/MoS2.yml` 为例），建议显式关闭自适应以避免混淆：

```yaml
separation:
  num_layers: 2
  method: separate_sum
  eta: 0.85
  sigma_y: 0.
  simplified: true
  adaptive_superposition: false   # 固定 k 时可关掉自适应
  fixed_superposition_k: 0.85     # 全局常数 k，优先级高于自适应
  residual_debug: false
```

若同时设置 `adaptive_superposition: true` 与 `fixed_superposition_k`，代码会优先采用固定模式（在残差调试 CSV 的 `k_mode` 字段可看到值为 `fixed`）。

### 残差调试模式（Residual Debug）
- 打开后记录每一步 `|residual_pos|` 的均值与自适应系数 `k`，用于排查收敛状态
- 支持 CSV + 折线图（PNG），输出路径与分解结果同目录，例如 `data/results/<mat>/<img>/<img>_residual_debug.*`

```yaml
separation:
  residual_debug: true  # 默认 false
```

CSV 字段：`iter, shift_h, shift_w, shift_index, t, t_next, lambda_t, k, k_mode, residual_abs_mean, adaptive`；自动生成的 `*_residual_debug.png` 横轴为迭代次数，纵轴同时绘制 `k`（蓝色）与 `|residual_pos|`（橙色）。

## 📁 项目结构

```
moire/
├── src/
│   ├── main.py                      # 分解主入口
│   ├── core/
│   │   ├── guided_diffusion/        # 扩散模型核心
│   │   ├── functions/               # DDNM 算子封装
│   │   └── scripts/                 # 训练/采样脚本
│   └── data_prep/                   # 仿真训练增强（online_augmentor.py）
├── configs/                         # 配置文件
│   ├── separate/*.yml               # 分解配置
│   ├── train/*.yml                  # 训练配置
│   ├── sample/*.yml                 # 采样配置
│   └── (训练增强配置合并至 train/*.yml)
├── data/
│   ├── multilayer/                  # 待分解图像（data/multilayer/<Material>/）
│   ├── results/                     # 分解结果
│   └── 仿真数据集/                   # 在线训练图像/掩膜
└── tools/                           # 实用工具
```

## 🧫 generate_sample：STEM 批量仿真（直接输出 PNG）

`generate_sample/` 是一个独立的小工作区，用于基于结构文件（`.xyz`）批量生成 ReS2 双层结构并调用 `incostem` 进行 STEM 仿真，**最终只输出 PNG**，不保留任何中间文件（中间的 `.xyz/.tif` 仅写入系统临时目录，进程结束会自动清理）。

### 依赖与可执行文件
- 需要 `generate_sample/incostem`（Linux 可执行文件）。源码在 `generate_sample/source/`，可在 `generate_sample/source/source/temsim` 下用 `make -f makefile.ubuntu incostem` 编译，然后拷贝到 `generate_sample/incostem` 并赋予可执行权限。
- 需要 Python 依赖：`ase`、`Pillow`（本项目环境一般已具备）。

### 配置与运行
- 编辑配置：`generate_sample/batch_config.json`（或直接使用 `generate_sample/ReS2.json`、`generate_sample/MoS2.json`、`generate_sample/MoTe2.json`、`generate_sample/TaS2.json`）
  - `output_dir`：最终 PNG 输出目录
  - `num`：最终输出图像数量；设置后会在给定范围内做随机均匀采样生成 `num` 张（此模式下 `x_range/y_range/rotation_range` 的 `step` 不参与枚举）
  - `pipeline_mode`：`final`（默认）或 `debug`；`debug` 会额外输出一份 `1644×1024` 的中间图到 `<output_dir>_debug1024/`，但仍在同一次运行里生成最终 `128×128`
  - `x_range / y_range`：滑移范围与步长（单位：Å）
  - `rotation_range`：扭角范围与步长（单位：度；若不需要扭角可删除该字段或设为 `null`）
  - `defect_rates_re`：Re 缺失率（0~1，小数；当前仅允许 1 个元素）
  - `mask_path`：裁剪安全 mask（默认 `generate_sample/mask/ReS2.png`）
  - `augment`：与 `configs/train/ReS2.yml:augment` 同构的增强配置（含 `disable`）
  - 网格枚举模式：如果不设置最外层 `num`，则按 `step` 进行网格枚举；当某个范围的 `step=0` 时，会在 `start` 与 `stop` 之间做等间距取值（可选用该范围内的 `num` 指定点数，含端点，不写则默认 `num=2`）
- 运行生成（一个入口即可，是否扭角由配置决定）：

```bash
python3 generate_sample/Batch_generate.py --config generate_sample/TaS2.json
```

### 可选：导出 GT 原子坐标（labels）
默认情况下，`generate_sample` **只输出 PNG**。如需在生成时同步保存 GT 原子坐标（用于对比分解结果是否出错），可通过环境变量开启：

```bash
# 生成 PNG，同时在 <output_dir>_labels/ 下保存每张图的 GT 坐标 npz 与对齐可视化
MOIRE_SAVE_LABELS=1 python3 generate_sample/Batch_generate.py --config generate_sample/TaS2.json
```

输出目录 `<output_dir>_labels/` 中每张样本对应：
- `<stem>.npz`：包含 `layer1_xy128/layer2_xy128`（128 坐标系 GT 原子点）、`gt_shift128/sideA/v1A/v2A/meta`
- `<stem>_gt_align.png`：将 `layer2 - gt_shift` 叠加到 `layer1` 的点云可视化（用于快速人工校验 GT）

其他可选环境变量：
- `MOIRE_TARGET_STEMS=a,b,c`：仅生成直到命中指定 stem 集合后提前停止（用于复现/补样）
- `MOIRE_GENERATE_VERBOSE=1`：打印每张样本的增强/GT 信息

### 输出文件命名约定
输出文件直接落在 `output_dir` 下，命名格式示例：
`ReS2_r0_gtx12.345_gty-6.789_sideA25.6.png`

- `ReS2`：材料名（来自配置 `material`）
- `r`：第二层绕 **+z 轴** 旋转角度（度；层间扭角 twist）。若不测扭角则为 `r0`
- `gtx/gty`：最终 `128×128` 像素坐标系下的滑移 GT（单位：像素；已同步经过增强链路中的 rotate/flip/resize）
- `sideA`：该 `128×128` patch 对应的物理边长（单位：Å）；因此 `Å/px = sideA / 128`
- 注意：构造端采样的 `x/y`（Å）与缺陷率 `d` 仍由 `batch_config.json` 控制，但不再写入文件名（避免与 eval 所需 GT 混杂）
- 输出分辨率：默认输出为增强后的 `128×128`；若 `pipeline_mode=debug`，会额外输出一份仿真中间图（约 `1644×1024`）

### 一键：分解 + eval（自动多 seed 重试）
传统流程是：先生成双层 PNG → 手动运行 `python src/main.py ...` 分解 → 再运行 eval 脚本。
现在可以直接用 eval 脚本完成“分解 + 原本 eval”，并在误差偏大时自动重试 4 次不同 seed，最终只保留误差最小的分解结果与 CSV 记录。

**滑移（shift_pbc）示例**（双层 PNG 在 `data/examples/slip_psnr24/MoS2/`，分解输出到 `outputs/separation/MoS2/` 后再评估）：
```bash
python tools/evaluate_generate_sample_wraparound_pbc.py \
  --task shift_pbc \
  --results_root outputs/separation/MoS2 \
  --batch_config generate_sample/MoS2.json \
  --separate_yml configs/separate/MoS2.yml \
  --separate_seed 42
```

**扭角（twist）示例**：
```bash
python tools/evaluate_generate_sample_wraparound_pbc.py \
  --task twist \
  --results_root outputs/separation/TaS2 \
  --batch_config generate_sample/TaS2.json \
  --separate_yml configs/separate/TaS2.yml \
  --separate_seed 42
```

**扭角（twist_labels，理想 GT 点云输入）示例**（直接使用 `MOIRE_SAVE_LABELS` 导出的 `.npz`）：
```bash
python tools/evaluate_generate_sample_wraparound_pbc.py \
  --task twist_labels \
  --results_root outputs/separation/TaS2_labels \
  --batch_config generate_sample/TaS2.json
```

说明：
- `--separate_yml` 会在进程启动时快照到内存（避免你并发实验中修改 yml 影响本次运行），并强制使用 `results_root.parent` 作为输入目录、`results_root` 作为输出目录。
- bad 阈值默认：`shift_pbc` 为 `--retry_thr_A=0.1`（Å），`twist` 为 `--retry_thr_deg=0.5`（度）；可按需调整。
- `twist` 的 `twist_period_deg`（用于按旋转对称性折叠角度）默认优先从 `--batch_config` 的 `twist_period_deg` 读取；也可用 `--twist_period_deg` 强制覆写。该值按材料/相而定：例如你当前约定为 MoS2=60、TaS2=120、MoTe2=180。
- 自动重试产生的中间结果、以及被替换掉的旧结果会移动到 `trash/eval_autopipeline_<task>_*/`，避免直接删除导致丢失。

## 🔬 ReS2 晶格基矢提取与层间物理量解析（pred_dadb + interlayer_analyzer_res2）

本项目在 `tools/pred_dadb/` 与 `src/tools/analysis/interlayer_analyzer_res2.py` 中实现了两项与 ReS2/HighT1 数据强相关的功能：

### 1) 单层晶格向量 (da, db) 提取 + 双层类别分类（slip/twist/flip_*）

入口脚本：`tools/pred_dadb/pipeline_bilayer_root.py`

功能：
- 对一个“包含多个双层分解子文件夹”的 root（例如 `data/HighT1/`）批处理。
- 每个子文件夹内读取两张单层图 `*_0.png` 与 `*_1.png`：
  - 提取 Re 原子点云（默认会落盘 atoms.json 以便复用/人工排查）
  - 在单层上估计原点 `origin` 与晶格基矢 `(da, db)`
  - 对双层样本输出粗类别：`slip / twist / flip_slip / flip_twist / unknown`

坐标系约定：
- 输入图片为 128×128；
- 输出的 `origin/da/db` 与落盘的 atoms.json **统一使用 512 坐标系**（默认 `out_scale=4`，即把 128 坐标系的 (x,y) 乘以 4）。

用法示例：
```bash
python tools/pred_dadb/pipeline_bilayer_root.py \
  --root data/HighT1 \
  --out_csv tools/pred_dadb/highT1_pred_dadb.csv
```

输出：
- `--out_csv`：每个双层样本一行，包含 `class`、两层的 `origin/da/db`（512 坐标系）、以及 `slip/flip_slip` 时的 `da_mean_512/db_mean_512` 等字段。
- `<out_csv_stem>_atoms/`：默认生成原子点云 json 与 `atoms_index.json`（点坐标为 512 坐标系）。

### 2) 基于 pred_dadb 输出的层间物理量解析（slip 平移 + twist 精细扭角）

入口脚本：`src/tools/analysis/interlayer_analyzer_res2.py`

功能（在 pred_dadb 粗分类基础上做“物理量”计算）：
- `slip`：在 512 坐标系下做粗到细平移搜索，得到 `shift_px`（layer0→layer1），并投影到 `(da,db)` 基底得到 `(c_a,c_b)`；同时输出 ReS2 规约后的 `(c_a,c_b)`。
- `twist`：用“位移向量集合匹配”的旋转搜索输出 `twist_fine_deg`（范围 [0,180)）与 `twist_score`。
- `flip_twist`：不做下游量计算。
- `flip_slip`：先把 layer0 的点云沿翻转轴反射（轴过 `origin0` 且垂直于 `db_mean`），转为等价的 non-flip slip 后再按 slip 流程计算。

它会优先复用 pred_dadb 的输出（若已存在则默认跳过 pipeline；可用 `--force_pred` 强制重跑）：
```bash
python src/tools/analysis/interlayer_analyzer_res2.py \
  --root data/HighT1 \
  --out_csv tools/pred_dadb/highT1_interlayer.csv \
  --pred_csv tools/pred_dadb/highT1_pred_dadb.csv \
  --pred_atoms_dir tools/pred_dadb/highT1_pred_dadb_atoms
```

输出：
- `--out_csv`：每个双层样本一行，包含 `class`、`shift_px/shift_A`、`ab_mean_reduced_*`（slip/flip_slip），以及 `twist_fine_deg`（twist）等字段。
  字段含义可直接参考脚本开头 docstring，或在 CSV 的列名上查看（示例文件：`tools/pred_dadb/highT1_interlayer_*.csv`）。

## 🔧 常用配置

### GPU 选择
**推荐方式**（避免脚本看到的GPU索引与nvidia-smi不一致）：
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python ...
```

**替代方式**（YAML 配置）：
```yaml
# 训练配置
train:
  gpu: 0  # 与 nvidia-smi 序号一致，-1 表示自动

# 采样/分解配置
runtime:
  gpu: 0
```

### 断点续训
在训练配置中设置：
```yaml
train:
  resume_checkpoint: models/checkpoints/ReS2/ema_0.9999_100000.pt
```

### 在线训练数据增强
```yaml
# 在 configs/train/ReS2.yml 中
augment:
  disable: []  # 可填入 ["noise", "carbon"] 等禁用某些增强

# 关闭保存样本避免 I/O 瓶颈
save_samples:
  enable: false
```

## 📊 训练日志分析

项目提供了训练日志分析工具：

```bash
# 基本分析（生成图表和统计报告）
python tools/analyze_training_loss.py

# 自定义参数
python tools/analyze_training_loss.py \
  --log_file models/checkpoints/ReS2/progress.csv \
  --breakpoint 50000 \
  --output analysis.png
```

**分析示例**（ReS2 100K→200K 断点续训）：
- VB损失激增 1743%（变分下界学习受冲击）
- 总损失跳跃 71%，但最终收敛到更低值
- MSE相对稳定，仅跳跃 27%

## 🗂️ 数据目录约定

- `data/multilayer/<Material>/`：待分解的多层图像（推理输入）
- `data/results/<Material>/<图片名>/`：分解输出
- `data/仿真数据集/`：在线训练的原始显微图与 mask（匹配 `train 配置中的数据生成段` 配置）
- `models/checkpoints/<model>/ema_*.pt`：规范化模型权重

## 📚 深入了解

- **[设计理念与架构](CLAUDE.md)**：了解项目的设计哲学、架构决策和扩展性考虑
  - YAML 驱动配置的设计权衡
  - 物理先验 vs 数据驱动的冲突解决
  - 核心约束与权衡
  - 版本演进历史
- **[配置详解](configs/)**：查看不同任务的配置模板
- **[API 文档](docs/)**：详细的技术文档和算法说明

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- [guided-diffusion](https://github.com/openai/guided-diffusion) - 扩散模型基础实现
- DDNM 论文 - 约束条件下的生成建模方法
- 材料科学社区 - 领域专业知识支持
