from pathlib import Path
from torch.utils.data import DataLoader
from src.data.liver_dataset import LiverSliceDataset

def create_dataloaders(
    split_dir: Path,
    batch_size,
    shuffle,
    num_workers
):
    split_dir = Path(split_dir)
    dataset = LiverSliceDataset(
        image_dir=split_dir / "images",
        mask_dir=split_dir / "masks"
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return loader

def create_train_val_test_loaders(
    processed_dir: Path,
    batch_size,
    num_workers
):
    processed_dir = Path(processed_dir)
    train_loader = create_dataloaders(
        split_dir=processed_dir / "train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = create_dataloaders(
        split_dir=processed_dir / "val",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = create_dataloaders(
        split_dir=processed_dir / "test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader, test_loader