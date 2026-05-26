# StackDiff

StackDiff is the official implementation of the paper **StackDiff: Human-like, physics-constrained unsupervised learning for picometer-accuracy layer-resolved stacking analysis**. It provides tools for STEM image layer decompostion, unconditional sampling, training configuration, simulated STEM data generation, symbolic regression for extracting image superposition formulas, ReS2 stacking analysis examples, and Re vacancy defect detection in monolayer ReS2.

This repository includes configurations and example inputs for various materials studied in the paper:

```text
ReS2, MoS2, MoTe2, TaS2...
```

Pretrained checkpoints and example training data will be available via Zenodo ([10.5281/zenodo.20375934](https://doi.org/10.5281/zenodo.20375935)).

## Overview

- `src/main.py`: Entry point for multilayer image separation based on YAML configuration files.
- `configs/separate/`: Image separation configurations for different materials.
- `configs/sample/`: Unconditional sampling configurations for different materials.
- `configs/train/`: Training configurations for different materials, using pregenerated monolayer/source STEM images with online augmentation during training.
- `data/examples/<material>/`: Small-scale demonstration inputs, with files named from `0.png` to `4.png`.
- `data/training_source/<material>/mask.png`: Mask required for training source image generation and online augmentation.
- `src/tools/synthetic_materials/`: Tools for simulated STEM image generation, including structural files and generation configurations for different materials. These tools are used to generate simulated data for training diffusion models of monolayer van der Waals materials.
- `src/tools/res2_stacking_analysis/`: ReS2-specific examples for stacking, slip, and twist analysis.
- `src/tools/Symbolic-regression/`: Symbolic regression tools for extracting intensity superposition formulas between monolayer and multilayer images from simulated data.
- `src/tools/monolayer_defect_detection/`: Model training and inference tools for Re vacancy defect detection in monolayer ReS2.


## Installation

We recommend installing the dependencies in an isolated Python environment:

```bash
pip install -r requirements.txt


## Model Weights

Each material uses a recommended EMA checkpoint by default. After downloading the weights, place them under the following paths:

```text
models/checkpoints/ReS2/ema_0.9999_200000.pt
models/checkpoints/MoS2/ema_0.9999_200000.pt
models/checkpoints/MoTe2/ema_0.9999_200000.pt
models/checkpoints/TaS2/ema_0.9999_200000.pt
...
```

Place the downloaded checkpoints in the paths listed above. The default configuration files already point to these locations.

After the weight download links are configured, the checkpoints can also be downloaded automatically using:

```bash
python scripts/download_weights.py --material all
```

## Image Decomposition

The following commands use the default example inputs. Run the commands below to separate the images in `data/examples/<material>/`:

```bash
python src/main.py --config configs/separate/ReS2.yml
python src/main.py --config configs/separate/MoS2.yml
python src/main.py --config configs/separate/MoTe2.yml
python src/main.py --config configs/separate/TaS2.yml
...
```

The default output directory is:

```text
outputs/separation/<material>/
```

Each input image corresponds to one output folder, which typically contains:

- `_original.png`：Preprocessed input image.
- `_0.png`：Separated layer 0.
- `_1.png`：Separated layer 1.
- `_combine.png`：Reconstructed overlap image of the two separated layers.

For custom inputs, modify the following fields in the corresponding YAML file:

```yaml
paths:
  default_input: ...
  default_output: ...
```

When auto-crop is enabled, if the filename contains a physical field-of-view suffix, such as `sample-7.1x3.1.png`, it indicates that the physical size of the image is 7.1 nm × 3.1 nm. The program will use this information to infer the crop and resize settings.

## Unconditional Sampling

After downloading the checkpoint for the corresponding material, unconditional sampling can be performed using:

```bash
python src/core/scripts/image_sample.py --config configs/sample/ReS2.yml
python src/core/scripts/image_sample.py --config configs/sample/MoS2.yml
python src/core/scripts/image_sample.py --config configs/sample/MoTe2.yml
python src/core/scripts/image_sample.py --config configs/sample/TaS2.yml
...
```

The sampling output directory is specified by `sample.output_dir` in `configs/sample/<material>.yml`.


## Training

Training uses pregenerated monolayer/source STEM simulation images and dynamically applies random augmentations during training. The training loader samples patches from the source images and applies augmentations such as cropping, rotation, elastic/perspective deformation, scan noise, and display transformation to generate 128 × 128 training patches.

The default source data directory is:

```text
data/training_source/<material>/
```

The repository already includes mask.png. Source images can be generated using the synthetic STEM data generation tools described below, or replaced with the user’s own simulation results.

Training can be launched with:

```bash
python src/core/scripts/image_train.py --config configs/train/ReS2.yml
python src/core/scripts/image_train.py --config configs/train/MoS2.yml
python src/core/scripts/image_train.py --config configs/train/MoTe2.yml
python src/core/scripts/image_train.py --config configs/train/TaS2.yml
```

Training checkpoints and logs will be written to the directory specified by `train.output_dir`.

## Simulated STEM Data Generation

`src/tools/synthetic_materials/` provides optional tools for simulated STEM data generation, including JSON, XYZ, and mask assets for multiple materials. If the required material system is not included, users can prepare their own assets.

This generation tool depends on the external executable `incostem` provided by the computem/temsim project:

```text
https://sourceforge.net/projects/computem/files/
```

`incostem` is not distributed with this repository. Place the local executable at:

```text
src/tools/synthetic_materials/incostem
```

Alternatively, set `incostem_path` in the material JSON file to an absolute path.

Generate monolayer/source STEM images for training:

```bash
python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/source_configs/ReS2.json
python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/source_configs/MoS2.json
python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/source_configs/MoTe2.json
python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/source_configs/TaS2.json
```

The source generator writes PNG images, `mask.png`, and `manifest.jsonl` under `data/training_source/<material>/`.

Generate synthetic bilayer PNG images for image separation examples or evaluation:

```bash
python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/ReS2.json
python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/MoS2.json
python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/MoTe2.json
python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/TaS2.json
```

## ReS2 Stacking Analysis Example

`src/tools/res2_stacking_analysis/` provides ReS2-specific examples for stacking analysis, including:

- Atom position detection.
- Monolayer lattice-vector estimation.
- da/db projection for slip samples.
- Classification of twist, slip, flip-twist, and flip-slip configurations.
- Visualization based on the analysis results.

This stacking analysis tool is currently designed only for ReS2. For other materials, this repository provides configurations for separation, sampling, training, and synthetic data generation.

The analysis tool requires each separated bilayer sample to be placed in an independent folder containing a pair of separated layer images:

```text
outputs/separation/ReS2/0/
  0_original.png
  0_0.png
  0_1.png
  0_combine.png
```

Here, `*_0.png` and `*_1.png` are required files.

The sample folder name may contain physical field-of-view information in the format `<width_nm>x<height_nm>`, for example:

```text
0_2.79x2.79/
```

When this type of tag is present, the tool converts the field-of-view size from nm to Angstrom. For example, `2.79 nm` is converted to `sideA_A = 27.9`. If no size tag is provided, the default value is:

```bash
--sideA_A 27.9
```

If the images correspond to different physical fields of view, this parameter should be manually overwritten.

Run stacking classification and monolayer lattice extraction:

```bash
python src/tools/res2_stacking_analysis/classify_bilayers.py \
  --root outputs/separation/ReS2 \
  --out_csv outputs/analysis/ReS2/res2_stacking.csv \
  --atoms_out_dir outputs/analysis/ReS2/atoms
```

Compute interlayer geometric quantities:

```bash
python src/tools/res2_stacking_analysis/analyze_interlayer.py \
  --root outputs/separation/ReS2 \
  --out_csv outputs/analysis/ReS2/interlayer.csv \
  --stacking_csv outputs/analysis/ReS2/res2_stacking.csv \
  --stacking_atoms_dir outputs/analysis/ReS2/atoms
```

For slip-type samples, the script outputs the pixel displacement, physical displacement, and its projection in the `(da, db)` basis. For twist-type samples, the script outputs the refined twist angle.

Alternatively, you can run `analyze_interlayer.py` alone and allow it to automatically invoke stacking classification when needed:

```bash
python src/tools/res2_stacking_analysis/analyze_interlayer.py \
  --root outputs/separation/ReS2 \
  --out_csv outputs/analysis/ReS2/interlayer.csv \
  --force_stacking
```

Generate analysis visualizations, including atom positions, monolayer lattice vectors, and slip/flip-slip alignment results:

```bash
python src/tools/res2_stacking_analysis/visualize_debug.py \
  --root outputs/separation/ReS2 \
  --stacking_csv outputs/analysis/ReS2/res2_stacking.csv \
  --atoms_dir outputs/analysis/ReS2/atoms \
  --interlayer_csv outputs/analysis/ReS2/interlayer.csv \
  --out_dir outputs/analysis/ReS2/visualization
```

## Symbolic Regression for Image Superposition

`src/tools/Symbolic-regression/` provides tools for extracting pixel-level intensity superposition formulas between monolayer and multilayer STEM images from simulated data. These tools are used to quantify how the gray value of a stacked multilayer image can be expressed as a symbolic function of the gray values of its constituent monolayer images.

This module contains two main notebooks:

- `Data_gen_for_SR.ipynb`: Generates simulated ADF-STEM image datasets for symbolic regression. It supports the generation of monolayer, bilayer, twist-stacked bilayer, and trilayer ReS2 image sets using incoSTEM simulations.
- `gray_regression_final.ipynb`: Extracts paired pixel gray values from monolayer and multilayer images and performs symbolic regression using PySR to obtain analytical intensity superposition formulas.

### Data Generation

`Data_gen_for_SR.ipynb` generates matched simulated STEM images for different stacking configurations. For bilayer ReS2, the generated database typically contains:

```text
DATABASE_PATH/
├── layer1/
│   └── data/
├── layer2/
│   └── data/
└── bilayer/
    └── data/
```

For trilayer ReS2, the generated database contains:

```text
DATABASE_PATH/
├── layer1/
│   └── data/
├── layer2/
│   └── data/
├── layer3/
│   └── data/
└── trilayer/
    └── data/
```

### Gray-value Extraction

`gray_regression_final.ipynb` extracts paired pixel gray values from matched monolayer and multilayer STEM images. For bilayer ReS2, each sampled pixel is saved as:

```text
x, y, layer1_gray, layer2_gray, bilayer_gray
```

For trilayer ReS2, each sampled pixel is saved as:

```text
x, y, layer1_gray, layer2_gray, layer3_gray, trilayer_gray
```

Valid pixels are selected from local atomic-intensity regions, and the extracted gray-value tables are saved as CSV files for subsequent symbolic regression.

### Symbolic Regression

`gray_regression_final.ipynb` performs symbolic regression using PySR to learn analytical intensity superposition formulas from the extracted gray-value data.

For bilayer ReS2, the regression target is:

```text
bilayer_gray = f(layer1_gray, layer2_gray)
```

For trilayer ReS2, the regression target is:

```text
trilayer_gray = f(layer1_gray, layer2_gray, layer3_gray)
```

The regression results include the selected symbolic expression, prediction metrics, predicted-versus-true plots, and exported Python predictor functions.

## Monolayer ReS2 Vacancy Defect Detection

`src/tools/monolayer_defect_detection/` provides training and inference tools for Re vacancy defect detection in monolayer ReS2 STEM images. The task is formulated as binary semantic segmentation of defect-related regions.

### Training

The model is trained using paired STEM images and binary masks. Training can be launched with:

```bash
python train_pt.py --config ReS2_vacancy_detect.yml
```
The configuration file controls the model architecture, encoder, input image size, batch size, learning rate, checkpoint saving, and early stopping settings.

### Inference

`Predict_visual_batch.ipynb` performs batch inference using a trained checkpoint. Before running the notebook, update the checkpoint path, input directory, and output directory:

```python
CKPT_PATH = "path/to/checkpoint.ckpt"
batch_root_dir = "path/to/input_images"
batch_save_dir = "path/to/output_results"
```

The notebook outputs predicted defect masks, visualization images, and summary files for detected Re vacancy defects.

## License

The code in this project is released under the MIT License. See `LICENSE` for details.

Licensing information for the datasets and model weights will be provided together with the Zenodo records.

## Acknowledgements

StackDiff builds on ideas from diffusion-based inverse-problem methods and the code organization style of OpenAI guided-diffusion. External simulation tools such as computem/temsim are independent projects and are not distributed with this repository.
