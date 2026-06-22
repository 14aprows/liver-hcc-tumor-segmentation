import matplotlib.pyplot as plt
from src.utils.file_utils import ensure_dir, build_timestamped_path

def plot_training_history(history, output_dir, prefix):
    output_dir = ensure_dir(output_dir)
    figure_path = build_timestamped_path(output_dir, prefix, ".png")

    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [row["train_dice"] for row in history], label="train dice")
    plt.plot(epochs, [row["val_dice"] for row in history], label="val dice")
    plt.plot(epochs, [row["train_iou"] for row in history], label="train iou")
    plt.plot(epochs, [row["val_iou"] for row in history], label="val iou")
    plt.title("Metrics")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(figure_path, dpi=150)
    plt.close()

    return figure_path