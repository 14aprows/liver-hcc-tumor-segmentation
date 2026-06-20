from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class LiverSliceDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(f"Image directory {self.image_dir} or mask directory {self.mask_dir} does not exist.")

        self.image_paths = sorted(self.image_dir.glob("*.npy"))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found in {self.image_dir}.")

        self.samples = []
        for image_path in self.image_paths:
            mask_path = self.mask_dir / image_path.name
            if mask_path.exists():
                self.samples.append({
                    "image_path": image_path,
                    "mask_path": mask_path,
                })
        if len(self.samples) == 0:
            raise RuntimeError(f"No matching image-mask pairs found in {self.image_dir} and {self.mask_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = np.load(sample["image_path"]).astype(np.float32)
        mask = np.load(sample["mask_path"]).astype(np.float32)

        if image.ndim != 2:
            raise ValueError(f"Image {sample['image_path']} has {image.ndim} dimensions; expected 2.")

        if mask.ndim != 2:
            raise ValueError(f"Mask {sample['mask_path']} has {mask.ndim} dimensions; expected 2.")

        if image.shape != mask.shape:
            raise ValueError(f"Image {sample['image_path']} and mask {sample['mask_path']} have different shapes: {image.shape} vs {mask.shape}.")

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask
        
