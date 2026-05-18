# Methods: Model Training

## Training data: simulation-to-experiment augmentation pipeline

StackDiff employs a purely simulation-based training strategy: no experimentally labelled data are required. A faithful monolayer STEM image prior is learned entirely from simulated images combined with online augmentation, and generalizes directly to experimental data.

Training data are generated online through a two-stage pipeline. First, the multislice simulation program *incostem* produces high-resolution monolayer HAADF-STEM images from crystal structure files, incorporating random vacancies and frozen-phonon atomic-position perturbations to capture realistic lattice disorder. Second, each simulated image is transformed on the fly during training by an online data augmentation pipeline comprising elastic deformation, projective transformation, amorphous carbon background, geometric augmentations (rotation, crop, flip), a physically motivated noise chain, and display-level adjustment, collectively bridging the domain gap between simulation and experiment (pipeline details and ablation study in Supplementary Note 2 and Supplementary Figs. 2–3).

## Model architecture

The denoising network is a U-Net operating on single-channel grayscale 128 × 128 images. Following the architecture of Dhariwal & Nichol, the network has 256 base channels with a channel multiplier of (1, 1, 2, 3, 4) across five resolution levels (maximum 1,024 channels), two residual blocks per resolution level, and four-head self-attention at 32 × 32, 16 × 16, and 8 × 8 feature-map resolutions. Time-step conditioning is injected via FiLM-based adaptive normalization (scale-shift norm). The network uses residual-block-based up/downsampling and predicts both noise and variance (learned sigma). The model comprises approximately 420 million parameters.

## Training objective

Training follows the standard denoising diffusion probabilistic model (DDPM) framework. In the forward process, Gaussian noise is progressively added to clean images over T = 1,000 time steps according to a linear variance schedule. The training objective is a simplified noise-prediction loss: at each uniformly sampled time step t, the U-Net learns to predict the noise ε added to the clean image, minimizing the mean squared error E[||ε − ε_θ(x_t, t)||²]. Because the model also predicts the variance, the actual training loss additionally includes a variational lower-bound term associated with the learned variance.

## Training setup

The model was trained for 200,000 steps with the AdamW optimizer (learning rate 1 × 10⁻⁴, weight decay 0) and a batch size of 32. Mixed-precision training (FP16) was used to improve throughput and reduce memory consumption. An exponential moving average (EMA) of the model weights (decay coefficient 0.9999) was maintained and used for inference. Training was performed on a single NVIDIA GPU. The online augmentation pipeline utilized 64 CPU worker processes for parallel sample generation, ensuring that GPU utilization was not bottlenecked by data loading. Checkpoints were saved every 10,000 steps.

At inference time, 200-step DDIM (Denoising Diffusion Implicit Models) accelerated sampling was used for efficient generation.
