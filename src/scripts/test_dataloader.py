from src.configs.config import (
    PROCESSED_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)

from src.data.liver_dataloaders import create_train_val_test_loaders

def main():
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")
    print(f"Test batches : {len(test_loader)}")

    images, masks = next(iter(train_loader))

    print()
    print("One train batch:")
    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Images dtype: {images.dtype}")
    print(f"Masks dtype: {masks.dtype}")
    print(f"Image min: {images.min().item()}")
    print(f"Image max: {images.max().item()}")
    print(f"Mask min: {masks.min().item()}")
    print(f"Mask max: {masks.max().item()}")

if __name__ == "__main__":
    main()