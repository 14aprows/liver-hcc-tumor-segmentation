from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_DIR / "nifti_and_segms"

BASELINE_IMAGE_FILE = "art.nii.gz"
BASELINE_MASK_FILE = "rater1_liver.nii.gz"

IMAGE_FILES = [
    "pre.nii.gz",
    "art.nii.gz",
    "pv.nii.gz",
    "del.nii.gz",
    "art_pre.nii.gz",
    "art_pv.nii.gz",
    "art_del.nii.gz",
]

LIVER_MASK_FILES = [
    "rater1_liver.nii.gz",
    "rater2_liver.nii.gz",
]

TUMOR_MASK_PATTERNS = [
    "rater1_tumor*.nii.gz",
    "rater2_tumor*.nii.gz",
]