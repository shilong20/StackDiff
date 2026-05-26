import os
import json
import glob
import torch
import random
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

from core.model import *
from core.data import *
from core.metrics import *


# =============================================================================
# Environment settings
# =============================================================================

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


# =============================================================================
# Segmentation model wrapper
# =============================================================================

class SegModel(nn.Module):
    """
    Segmentation model wrapper.

    The model architecture, encoder, weights and input channels are controlled
    by the YAML configuration file.
    """

    def __init__(self, model_cfg):
        super().__init__()

        self.model = SMPModelFactory(
            in_channels=model_cfg["in_channels"]
        ).get_model(
            arch_name=model_cfg["arch"],
            encoder_name=model_cfg["encoder"],
            encoder_weights=model_cfg["weights"],
        )

        self.loss_fn = CustomLoss()

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


# =============================================================================
# Checkpoint manager
# =============================================================================

class CheckpointManager:
    """
    Save and manage the top-k best checkpoints according to a monitored metric.
    """

    def __init__(self, save_dir, top_k=10, mode="max", monitor="val_dice"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.top_k = top_k
        self.mode = mode
        self.monitor = monitor
        self.best_scores = []

    def _is_better(self, score, best_score):
        if self.mode == "max":
            return score > best_score
        return score < best_score

    def save(self, model, optimizer, scheduler, epoch, metrics):
        """
        Save one checkpoint and keep only the top-k best checkpoints.
        """
        score = metrics.get(self.monitor, 0)

        val_loss = metrics.get("val_loss", 0)
        val_dice = metrics.get("val_dice", 0)

        filename = f"epoch={epoch}-val_loss={val_loss:.4f}-val_dice={val_dice:.4f}.ckpt"
        filepath = self.save_dir / filename

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        }

        torch.save(checkpoint, filepath)

        self.best_scores.append((score, filepath))
        self.best_scores.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

        while len(self.best_scores) > self.top_k:
            _, old_path = self.best_scores.pop()
            if old_path.exists():
                old_path.unlink()

        print(f"  Saved checkpoint: {filename}")
        return filepath

    def get_best_checkpoint(self):
        """
        Return the best checkpoint path.
        """
        if self.best_scores:
            return self.best_scores[0][1]
        return None

    def load(self, filepath, model, optimizer=None, scheduler=None):
        """
        Load model, optimizer and scheduler states from a checkpoint.
        """
        checkpoint = torch.load(filepath, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})


# =============================================================================
# Early stopping
# =============================================================================

class EarlyStopping:
    """
    Stop training when the monitored score does not improve for a given patience.
    """

    def __init__(self, patience=20, mode="max", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta

        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def _is_improvement(self, score):
        if self.best_score is None:
            return True

        if self.mode == "max":
            return score > self.best_score + self.min_delta

        return score < self.best_score - self.min_delta

    def step(self, score):
        """
        Update early-stopping state with the current monitored score.
        """
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\nEarly stopping triggered: no improvement for {self.patience} epochs")

        return self.should_stop


# =============================================================================
# Binary visualization callback
# =============================================================================

class BinaryVisualizationCallback:
    """
    Visualization callback for binary ReS2 atom segmentation.

    Class definition:
    - 0: background
    - 1: ReS2 atom
    """

    def __init__(self, log_dir="logs", num_samples=1):
        self.log_dir = log_dir
        self.num_samples = num_samples

        self.class_colors = {
            0: [0, 0, 0],
            1: [255, 0, 0],
        }

        self.class_names = {
            0: "Background",
            1: "ReS2 Atom",
        }

    def label_to_color(self, label_img, sample_type, sample_idx):
        """
        Convert a single-channel binary label map into an RGB color map.
        """
        label_img = label_img.astype(np.int32)
        label_img = np.clip(label_img, 0, 1)

        h, w = label_img.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id in [1, 0]:
            color = self.class_colors[class_id]
            mask = label_img == class_id

            color_img[mask, 0] = color[0]
            color_img[mask, 1] = color[1]
            color_img[mask, 2] = color[2]

        return color_img

    def visualize_sample(self, image, label, pred, epoch, sample_type, idx):
        """
        Save one visualization panel containing input image, ground truth and prediction.
        """
        if image.shape != label.shape:
            label = np.resize(label, image.shape)

        if isinstance(pred, torch.Tensor):
            pred_class = torch.argmax(pred, dim=0).cpu().numpy()
        else:
            pred_class = np.argmax(pred, axis=0)

        pred_class = pred_class.astype(np.int32)
        pred_class = np.clip(pred_class, 0, 1)

        label_color = self.label_to_color(label, sample_type, idx)
        pred_color = self.label_to_color(pred_class, f"{sample_type}_pred", idx)

        image_vis = image.copy()

        if image_vis.dtype == np.float32:
            image_vis = (image_vis * 255).astype(np.uint8)

        image_vis = np.clip(image_vis, 0, 255)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(
            f"Epoch {epoch} - {sample_type.upper()} Sample {idx}",
            fontsize=12,
            fontweight="bold",
        )

        axes[0].imshow(image_vis, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Input Image")
        axes[0].axis("off")

        axes[1].imshow(label_color)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_color)
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(
                facecolor=np.array(self.class_colors[0]) / 255,
                label=self.class_names[0],
            ),
            Patch(
                facecolor=np.array(self.class_colors[1]) / 255,
                label=self.class_names[1],
            ),
        ]

        fig.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, -0.08),
            frameon=False,
        )

        plt.tight_layout()

        save_dir = Path(self.log_dir) / "binary_visualizations"
        save_dir.mkdir(exist_ok=True, parents=True)

        save_path = save_dir / f"epoch{epoch:03d}_{sample_type}_sample{idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

    def on_epoch_end(self, model, train_loader, val_loader, epoch, device):
        """
        Visualize one random training sample and one random validation sample.
        """
        model.eval()

        try:
            random_batch_idx = random.randint(0, len(train_loader) - 1)

            for idx, batch in enumerate(train_loader):
                if idx == random_batch_idx:
                    images, labels = batch
                    random_sample_idx = random.randint(0, images.shape[0] - 1)

                    image = images[random_sample_idx].cpu().numpy().squeeze()
                    label = labels[random_sample_idx].cpu().numpy()

                    with torch.no_grad():
                        pred = model(images[random_sample_idx:random_sample_idx + 1].to(device))
                        pred = pred.squeeze(0)

                    self.visualize_sample(
                        image=image,
                        label=label,
                        pred=pred,
                        epoch=epoch,
                        sample_type="train",
                        idx=random_sample_idx,
                    )
                    break

        except Exception as exc:
            print(f"Training visualization failed: {exc}")

        try:
            random_batch_idx = random.randint(0, len(val_loader) - 1)

            for idx, batch in enumerate(val_loader):
                if idx == random_batch_idx:
                    images, labels = batch
                    random_sample_idx = random.randint(0, images.shape[0] - 1)

                    image = images[random_sample_idx].cpu().numpy().squeeze()
                    label = labels[random_sample_idx].cpu().numpy().astype(np.int32)

                    with torch.no_grad():
                        pred = model(images[random_sample_idx:random_sample_idx + 1].to(device))
                        pred = pred.squeeze(0)

                    self.visualize_sample(
                        image=image,
                        label=label,
                        pred=pred,
                        epoch=epoch,
                        sample_type="valid",
                        idx=random_sample_idx,
                    )
                    break

        except Exception as exc:
            print(f"Validation visualization failed: {exc}")


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """
    Pure PyTorch trainer for semantic segmentation.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        log_dir,
        max_epochs=1000,
        checkpoint_manager=None,
        early_stopping=None,
        visualization_callback=None,
    ):
        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.log_dir = Path(log_dir)
        self.max_epochs = max_epochs

        self.checkpoint_manager = checkpoint_manager
        self.early_stopping = early_stopping
        self.visualization_callback = visualization_callback

        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.current_epoch = 0

    def train_one_epoch(self):
        """
        Train the model for one epoch with a tqdm progress bar.
        """
        self.model.train()

        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Training Epoch {self.current_epoch + 1}/{self.max_epochs}",
            unit="batch",
            colour="green",
        )

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            preds = self.model(images)
            loss = self.model.compute_loss(preds, labels)

            loss.backward()
            self.optimizer.step()

            batch_iou = iou(preds, labels)
            batch_dice = dice(preds, labels)

            b_loss = loss.item()
            b_iou = batch_iou.item() if isinstance(batch_iou, torch.Tensor) else batch_iou
            b_dice = batch_dice.item() if isinstance(batch_dice, torch.Tensor) else batch_dice

            total_loss += b_loss
            total_iou += b_iou
            total_dice += b_dice
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{b_loss:.4f}",
                "dice": f"{b_dice:.4f}",
            })

        return {
            "train_loss": total_loss / num_batches,
            "train_iou": total_iou / num_batches,
            "train_dice": total_dice / num_batches,
        }

    @torch.no_grad()
    def validate(self):
        """
        Evaluate the model on the validation set with a tqdm progress bar.
        """
        self.model.eval()

        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        num_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc="Validating",
            unit="batch",
            leave=False,
            colour="blue",
        )

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            preds = self.model(images)
            loss = self.model.compute_loss(preds, labels)

            batch_iou = iou(preds, labels)
            batch_dice = dice(preds, labels)

            b_loss = loss.item()
            b_iou = batch_iou.item() if isinstance(batch_iou, torch.Tensor) else batch_iou
            b_dice = batch_dice.item() if isinstance(batch_dice, torch.Tensor) else batch_dice

            total_loss += b_loss
            total_iou += b_iou
            total_dice += b_dice
            num_batches += 1

            pbar.set_postfix({
                "val_dice": f"{b_dice:.4f}",
            })

        return {
            "val_loss": total_loss / num_batches,
            "val_iou": total_iou / num_batches,
            "val_dice": total_dice / num_batches,
        }

    def fit(self, resume_from=None):
        """
        Run the full training process.
        """
        start_epoch = 0

        if resume_from and Path(resume_from).exists():
            print(f"Resume from checkpoint: {resume_from}")

            start_epoch, _ = self.checkpoint_manager.load(
                resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
            )
            start_epoch += 1

        print(f"\n{'=' * 60}")
        print(f"Start training: {self.max_epochs} epochs | Experiment: {self.log_dir.name}")
        print(f"{'=' * 60}\n")

        for epoch in range(start_epoch, self.max_epochs):
            self.current_epoch = epoch

            train_metrics = self.train_one_epoch()
            val_metrics = self.validate()

            metrics = {**train_metrics, **val_metrics}

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(metrics["val_dice"])
            new_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1} Summary: "
                f"Train Dice: {metrics['train_dice']:.4f} | "
                f"Val Dice: {metrics['val_dice']:.4f} | "
                f"Val Loss: {metrics['val_loss']:.4f} | "
                f"LR: {new_lr:.2e}"
            )

            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)
            self.writer.add_scalar("learning_rate", new_lr, epoch)

            if self.checkpoint_manager:
                self.checkpoint_manager.save(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    metrics,
                )

            if self.visualization_callback:
                self.visualization_callback.on_epoch_end(
                    self.model,
                    self.train_loader,
                    self.val_loader,
                    epoch,
                    self.device,
                )

            if self.early_stopping:
                if self.early_stopping.step(metrics["val_dice"]):
                    print(f"\nTraining completed with early stopping: {epoch + 1} epochs")
                    break

        else:
            print(f"\nTraining completed: {self.max_epochs} epochs")

        self.writer.close()

        if self.checkpoint_manager:
            return self.checkpoint_manager.get_best_checkpoint()

        return None

    @torch.no_grad()
    def predict(self, dataloader, ckpt_path=None):
        """
        Run inference on a dataloader.
        """
        if ckpt_path == "best" and self.checkpoint_manager:
            best_ckpt = self.checkpoint_manager.get_best_checkpoint()

            if best_ckpt:
                print(f"Load best checkpoint: {best_ckpt}")
                self.checkpoint_manager.load(best_ckpt, self.model)

        elif ckpt_path and Path(ckpt_path).exists():
            print(f"Load checkpoint: {ckpt_path}")

            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.model.eval()
        predictions = []

        print("Running inference...")

        for images, labels in tqdm(dataloader, desc="Inference", unit="batch"):
            images = images.to(self.device)
            preds = self.model(images)
            predictions.append((preds.cpu(), labels))

        return predictions


# =============================================================================
# Utility functions
# =============================================================================

def get_version_dir(base_dir, name):
    """
    Create and return an auto-incremented version directory.
    """
    base_path = Path(base_dir) / name

    version = 0
    while (base_path / f"version_{version}").exists():
        version += 1

    version_dir = base_path / f"version_{version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    return version_dir


# =============================================================================
# Main entry
# =============================================================================

if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    # Load YAML configuration.
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"Loading configuration: {args.config}")
    print(f"Experiment name: {cfg['experiment_name']}")

    # Set random seeds.
    seed = cfg["system"].get("seed", 42)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Prepare image paths.
    data_dir = os.path.join(
        cfg["data"]["data_root"],
        cfg["data"]["dataset_name"],
    )

    train_img_list = np.array(
        glob.glob(os.path.join(data_dir, "train", "*.jpg"))
    ).tolist()

    valid_img_list = np.array(
        glob.glob(os.path.join(data_dir, "valid", "*.jpg"))
    ).tolist()

    print(f"Train nums: {len(train_img_list)}, Valid nums: {len(valid_img_list)}")

    # Build datasets and dataloaders.
    bs = cfg["data"]["batch_size"]
    nw = cfg["system"]["num_workers"]
    dim = cfg["data"]["image_size"]

    train_dataset = CustomDataset(
        train_img_list,
        dim=dim,
        data_type="train",
    )

    valid_dataset = CustomDataset(
        valid_img_list,
        dim=dim,
        data_type="valid",
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=bs,
        num_workers=nw,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
    )

    # Select device and build model.
    device = torch.device(
        f"cuda:{cfg['system'].get('gpus', 0)}"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    model = SegModel(model_cfg=cfg["model"])

    # Build optimizer and learning-rate scheduler.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=cfg["train"]["scheduler_factor"],
        mode="max",
        patience=2,
        min_lr=0,
    )

    # Create log directory.
    results_root = "/hwq/seg/code/results"
    log_dir = get_version_dir(results_root, cfg["experiment_name"])

    print(f"Log directory: {log_dir}")

    # Back up the current configuration file.
    with open(log_dir / "config_backup.yml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    # Initialize training utilities.
    ckpt_dir = log_dir / "checkpoints"

    checkpoint_manager = CheckpointManager(
        save_dir=ckpt_dir,
        top_k=cfg["train"]["save_top_k"],
        mode="max",
        monitor="val_dice",
    )

    early_stopping = EarlyStopping(
        patience=cfg["train"]["early_stop_patience"],
        mode="max",
        min_delta=0.0,
    )

    visualization_callback = BinaryVisualizationCallback(
        log_dir=str(log_dir),
        num_samples=1,
    )

    # Create trainer.
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=str(log_dir),
        max_epochs=cfg["train"]["epochs"],
        checkpoint_manager=checkpoint_manager,
        early_stopping=early_stopping,
        visualization_callback=visualization_callback,
    )

    # Start training.
    # To resume training from a checkpoint:
    # ckpt_path = r"..."
    # trainer.fit(resume_from=ckpt_path)

    best_ckpt = trainer.fit()

    # Run inference on the validation set.
    print("\nStart inference...")

    predictions = trainer.predict(
        dataloader=valid_loader,
        ckpt_path="best",
    )

    # Save validation predictions and labels.
    preds = torch.squeeze(
        torch.concat([item[0] for item in predictions])
    ).numpy().tolist()

    labels = torch.squeeze(
        torch.concat([item[1] for item in predictions])
    ).numpy().tolist()

    results = {
        "img_path": valid_img_list,
        "pred": preds,
        "label": labels,
    }

    results_json = json.dumps(results)

    results_path = log_dir / "valid.json"

    with open(results_path, "w+", encoding="utf-8") as f:
        f.write(results_json)

    print(f"\nResults saved to: {results_path}")