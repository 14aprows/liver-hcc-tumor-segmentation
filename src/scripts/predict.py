import matplotlib.pyplot as plt
import numpy as np
import torch

from src.configs.config import (
    PROCESSED_DIR,
    CHECKPOINTS_DIR,
    PREDICTIONS_DIR,
    FIGURES_DIR,
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
)
from src.data.liver_dataloaders import create_train_val_test_loaders
from src.models.unet import UNet
from src.models.resunet import ResUNet
from src.models.attention_unet import AttentionUNet
from src.utils.file_utils import ensure_dir

def resolve_device():
    if DEVICE == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(DEVICE)

def save_prediction_figure(image, mask, pred, output_path):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap="gray")
    plt.imshow(pred, cmap="Reds", alpha=0.35)
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def predict_model(model_class, test_loader, device, threshold=0.5, max_samples=16):
    model = model_class(in_channels=1, out_channels=1).to(device)
    model_name = model.__class__.__name__.lower()

    checkpoint_path = CHECKPOINTS_DIR / f"best_{model_name}.pth"

    if not checkpoint_path.exists():
        print(f"Skipping {model_name}: checkpoint not found: {checkpoint_path}")
        return

    prediction_dir = ensure_dir(PREDICTIONS_DIR / model_name)
    figure_dir = ensure_dir(FIGURES_DIR / "predictions" / model_name)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    saved = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()
            preds_np = preds.cpu().numpy()

            for i in range(images_np.shape[0]):
                image = images_np[i, 0]
                mask = masks_np[i, 0]
                pred = preds_np[i, 0]

                npy_path = prediction_dir / f"{model_name}_pred_{saved:04d}.npy"
                fig_path = figure_dir / f"{model_name}_pred_{saved:04d}.png"

                np.save(npy_path, pred.astype(np.float32))
                save_prediction_figure(image, mask, pred, fig_path)

                saved += 1

                if saved >= max_samples:
                    print(f"Saved {model_name} predictions: {saved}")
                    print(f"Prediction dir: {prediction_dir}")
                    print(f"Figure dir: {figure_dir}")
                    return

def main():
    device = resolve_device()

    _, _, test_loader = create_train_val_test_loaders(
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model_classes = [
        UNet,
        ResUNet,
        AttentionUNet,
    ]

    for model_class in model_classes:
        predict_model(
            model_class=model_class,
            test_loader=test_loader,
            device=device,
        )

if __name__ == "__main__":
    main()