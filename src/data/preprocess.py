import hashlib
import re
from pathlib import Path
import cv2
import numpy as np

from src.utils.file_utils import ensure_dir
from src.utils.nifti_utils import load_nifti_array

def sanitize_name(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text)
    text = text.strip("_")
    return text

def make_short_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]

def normalize_ct_slice(
    image_slice,
    hu_min,
    hu_max,
):
    image_slice = np.clip(image_slice, hu_min, hu_max)
    image_slice = (image_slice - hu_min) / (hu_max - hu_min)
    return image_slice.astype(np.float32)

def resize_image(
    image_slice,
    image_size,
):
    return cv2.resize(
        image_slice,
        image_size,
        interpolation=cv2.INTER_LINEAR,
    )

def resize_mask(
    mask_slice: np.ndarray,
    image_size,
):
    return cv2.resize(
        mask_slice,
        image_size,
        interpolation=cv2.INTER_NEAREST,
    )

def preprocess_single_study(
    case_id,
    study_id,
    image_path: Path,
    mask_path: Path,
    output_image_dir: Path,
    output_mask_dir: Path,
    image_size,
    hu_min,
    hu_max,
    skip_empty_mask,
):
    image_volume = load_nifti_array(image_path)
    mask_volume = load_nifti_array(mask_path)

    if image_volume.shape != mask_volume.shape:
        raise ValueError(
            f"Shape mismatch for {case_id}/{study_id}: "
            f"image={image_volume.shape}, mask={mask_volume.shape}"
        )

    if image_volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume for {case_id}/{study_id}, "
            f"got shape={image_volume.shape}"
        )

    ensure_dir(output_image_dir)
    ensure_dir(output_mask_dir)

    safe_case_id = sanitize_name(case_id)
    safe_study_id = make_short_id(study_id)

    saved_slices = 0
    depth = image_volume.shape[2]

    for z in range(depth):
        image_slice = image_volume[:, :, z]
        mask_slice = mask_volume[:, :, z]

        mask_slice = (mask_slice > 0).astype(np.float32)

        if skip_empty_mask and mask_slice.sum() == 0:
            continue

        image_slice = normalize_ct_slice(
            image_slice=image_slice,
            hu_min=hu_min,
            hu_max=hu_max,
        )

        image_slice = resize_image(
            image_slice=image_slice,
            image_size=image_size,
        )

        mask_slice = resize_mask(
            mask_slice=mask_slice,
            image_size=image_size,
        )

        filename = f"{safe_case_id}_{safe_study_id}_slice_{z:03d}.npy"

        np.save(
            output_image_dir / filename,
            image_slice.astype(np.float32),
        )

        np.save(
            output_mask_dir / filename,
            mask_slice.astype(np.float32),
        )

        saved_slices += 1

    return saved_slices