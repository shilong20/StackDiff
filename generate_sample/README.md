# generate_sample

`generate_sample/` 用于生成四种材料的双层 STEM 仿真 PNG：`ReS2`、`MoS2`、`MoTe2`、`TaS2`。公开仓库只保留生成所需的脚本、材料配置、结构文件和 mask；中间 `.xyz/.tif` 会写到临时目录，默认只落盘最终 PNG。

## 外部依赖

`incostem` 不随本仓库分发。请从官方 computem/temsim 项目下载或编译：

https://sourceforge.net/projects/computem/files/

然后选择一种方式让配置找到可执行文件：

```bash
# 推荐：放到本目录下，材料 JSON 里的默认 incostem_path 可直接使用
cp /path/to/incostem generate_sample/incostem
chmod +x generate_sample/incostem
```

也可以直接编辑材料 JSON 中的 `incostem_path`，填入本机的绝对路径或相对路径。`generate_sample/incostem` 已加入 `.gitignore`，本地放置后不会误提交。

## 目录内容

- `Batch_generate.py`：命令行入口。
- `batch_runner.py`：生成双层结构、调用 `incostem`、增强并输出 PNG 的核心实现。
- `ReS2.json`、`MoS2.json`、`MoTe2.json`、`TaS2.json`：四种材料的生成配置。
- `ReS2.xyz`、`MoS2.xyz`、`MoTe2.xyz`、`TaS2.xyz`：四种材料的单层结构。
- `mask/*.png`：四种材料对应的安全裁剪 mask。

## 运行示例

```bash
python generate_sample/Batch_generate.py --config generate_sample/ReS2.json
python generate_sample/Batch_generate.py --config generate_sample/MoS2.json
python generate_sample/Batch_generate.py --config generate_sample/MoTe2.json
python generate_sample/Batch_generate.py --config generate_sample/TaS2.json
```

主要配置字段：

- `output_dir`：最终 PNG 输出目录。
- `num`：输出图像数量；设置后会在范围内随机采样。
- `structure_path`：当前材料的 `.xyz` 文件。
- `incostem_path`：本机 `incostem` 可执行文件路径。
- `mask_path`：当前材料的裁剪 mask。
- `x_range` / `y_range`：层间滑移采样范围，单位为 Å。
- `rotation_range`：扭角范围，单位为度。
- `augment`：生成后的图像增强配置。

## 可选 labels

默认只输出 PNG。如需同步保存 GT 原子坐标：

```bash
MOIRE_SAVE_LABELS=1 python generate_sample/Batch_generate.py --config generate_sample/ReS2.json
```

labels 会保存到 `<output_dir>_labels/`，用于调试或评估滑移、扭角等 GT。
