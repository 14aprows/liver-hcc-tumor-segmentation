from pathlib import Path
import nibabel as nib
import numpy as np

def load_nifti(path: Path):
    path = Path(path)
    return nib.load(str(path))

def load_nifti_array(path: Path):
    path = Path(path)
    return nib.load(str(path)).get_fdata(dtype=np.float32)

def load_nifti_shape(path: Path):
    try:
        return load_nifti(path).shape
    except Exception as error:
        return f"Error loading {path}: {error}"

def read_mask_info(path: Path):
    try:
        mask = load_nifti_array(path)
        return {
            "shape": mask.shape,
            "unique_values": np.unique(mask).tolist(),
            "positive_voxels": int((mask > 0).sum()),
        }
    except Exception as error:
        return f"Error reading mask info for {path}: {error}"