import torch

from src.configs.config import (
    PROCESSED_DIR,
    CHECKPOINTS_DIR,
    TRAINING_LOGS_DIR,
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
)
from src.data.liver_dataloaders import create_train_val_test_loaders
from src.models.unet import UNet
from src.training.losses import BCEDiceLoss
from src.training.trainer import train_one_epoch, evaluate
from src.utils.file_utils import ensure_dir, build_timestamped_path, write_rows_to_csv

def resolve_device():
    if DEVICE == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")

    return torch.device(DEVICE)

def main():
    ensure_dir(CHECKPOINTS_DIR)
    ensure_dir(TRAINING_LOGS_DIR)

    device = resolve_device()
    print(f"Device: {device}")

    train_loader, val_loader, _ = create_train_val_test_loaders(
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_dice = 0.0
    history = []

    best_checkpoint_path = CHECKPOINTS_DIR / "best_unet.pth"
    last_checkpoint_path = CHECKPOINTS_DIR / "last_unet.pth"

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        }
        history.append(row)

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_metrics['loss']:.4f} "
            f"Train Dice: {train_metrics['dice']:.4f} "
            f"Train IoU: {train_metrics['iou']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} "
            f"Val Dice: {val_metrics['dice']:.4f} "
            f"Val IoU: {val_metrics['iou']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                best_checkpoint_path,
            )

            print(f"Saved best model: {best_checkpoint_path}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
            last_checkpoint_path,
        )

    history_path = build_timestamped_path(
        output_dir=TRAINING_LOGS_DIR,
        prefix="train_history",
        suffix=".csv",
    )

    fieldnames = [
        "epoch",
        "train_loss",
        "train_dice",
        "train_iou",
        "val_loss",
        "val_dice",
        "val_iou",
    ]

    write_rows_to_csv(
        path=history_path,
        rows=history,
        fieldnames=fieldnames,
    )

    print()
    print("Training finished.")
    print(f"Best val Dice: {best_val_dice:.4f}")
    print(f"Best model: {best_checkpoint_path}")
    print(f"Last model: {last_checkpoint_path}")
    print(f"Training log: {history_path}")

if __name__ == "__main__":
    main()