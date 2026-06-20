from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATASET_DIR / "nifti_and_segms"

PROCESSED_DIR = PROJECT_ROOT / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR = OUTPUTS_DIR / "figures"

LOGS_DIR = PROJECT_ROOT / "logs"
DATASET_CHECK_LOGS_DIR = LOGS_DIR / "dataset_checks"
TRAINING_LOGS_DIR = LOGS_DIR / "training"
EVALUATION_LOGS_DIR = LOGS_DIR / "evaluation"

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

IMAGE_SIZE = (256, 256)
HU_MIN = -100
HU_MAX = 400
SKIP_EMPTY_MASK = True
RESET_PROCESSED_DIR = True

RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 8
NUM_WORKERS = 0