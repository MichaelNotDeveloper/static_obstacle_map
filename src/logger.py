import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


class Logger:
    def __init__(
        self,
        models,
        output_dir="runs/train",
        monitor="val_iou",
        mode="max",
    ):
        self.models = models if isinstance(models, dict) else {"model": models}
        self.output_dir = Path(output_dir)
        self.plot_dir = self.output_dir / "plots"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.best_metric = -float("inf") if mode == "max" else float("inf")
        self.is_better = (lambda new, old: new > old) if mode == "max" else (
            lambda new, old: new < old
        )

        self.history = []
        self.reset_epoch()

    def reset_epoch(self):
        self.train_losses = []
        self.val_losses = []
        self.train_scores = {}
        self.val_scores = {}
        self.sample = None

    def log_train_loss(self, value):
        self.train_losses.append(float(value))

    def log_val_loss(self, value):
        self.val_losses.append(float(value))

    def log_train_score(self, score):
        self._log_score(self.train_scores, score)

    def log_val_score(self, score):
        self._log_score(self.val_scores, score)

    def add_sample(self, logits, target):
        if target.ndim == 3:
            target = target.unsqueeze(1)
        self.sample = {
            "prob": torch.sigmoid(logits.detach().cpu()),
            "target": target.detach().cpu(),
        }

    def update_batch(self, epoch=None):
        epoch = len(self.history) + 1 if epoch is None else int(epoch)
        metrics = {"epoch": epoch}
        metrics["train_loss"] = self._mean(self.train_losses)
        metrics["val_loss"] = self._mean(self.val_losses)
        metrics.update(self._mean_scores("train", self.train_scores))
        metrics.update(self._mean_scores("val", self.val_scores))

        self.history.append(metrics)
        self._print_metrics(metrics)
        self._save_history()
        self._save_plot(epoch)
        self._save_best_model(epoch, metrics)
        self.reset_epoch()

    @staticmethod
    def _log_score(storage, score):
        if isinstance(score, dict):
            for name, value in score.items():
                storage.setdefault(name, []).append(float(value))
        else:
            storage.setdefault("iou", []).append(float(score))

    @staticmethod
    def _mean(values):
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _mean_scores(self, prefix, storage):
        return {
            f"{prefix}_{name}": self._mean(values)
            for name, values in sorted(storage.items())
        }

    def _print_metrics(self, metrics):
        text = [f"epoch={metrics['epoch']}"]
        for name, value in metrics.items():
            if name == "epoch":
                continue
            text.append(f"{name}={value:.5f}")
        print(" | ".join(text))

    def _save_history(self):
        history_path = self.output_dir / "metrics.json"
        with history_path.open("w") as f:
            json.dump(self.history, f, indent=2)

    def _save_best_model(self, epoch, metrics):
        metric = metrics.get(self.monitor)
        if metric is None or not self.is_better(metric, self.best_metric):
            return

        self.best_metric = metric
        checkpoint = {
            "epoch": epoch,
            "best_metric": self.best_metric,
            "monitor": self.monitor,
            "models": {
                name: model.state_dict()
                for name, model in self.models.items()
                if hasattr(model, "state_dict")
            },
            "history": self.history,
        }
        torch.save(checkpoint, self.checkpoint_dir / "best.pt")

    def _save_plot(self, epoch):
        if not self.history:
            return

        epochs = [item["epoch"] for item in self.history]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        axes[0, 0].plot(epochs, [item["train_loss"] for item in self.history], label="train")
        axes[0, 0].plot(epochs, [item["val_loss"] for item in self.history], label="val")
        axes[0, 0].set_title("Loss")
        axes[0, 0].grid(True)
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, [item.get("train_iou", 0.0) for item in self.history], label="train")
        axes[0, 1].plot(epochs, [item.get("val_iou", 0.0) for item in self.history], label="val")
        axes[0, 1].set_title("IoU")
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        self._draw_sample(axes[1, 0], axes[1, 1])

        fig.tight_layout()
        fig.savefig(self.plot_dir / f"epoch_{epoch:04d}.png", dpi=160)
        plt.close(fig)

    def _draw_sample(self, pred_ax, target_ax):
        pred_ax.axis("off")
        target_ax.axis("off")
        pred_ax.set_title("Prediction")
        target_ax.set_title("Target")

        if self.sample is None:
            return

        prob = self.sample["prob"]
        target = self.sample["target"]
        sample_id = torch.randint(0, prob.shape[0], (1,)).item()
        pred_ax.imshow(prob[sample_id, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        target_ax.imshow(target[sample_id, 0].numpy(), cmap="viridis")
