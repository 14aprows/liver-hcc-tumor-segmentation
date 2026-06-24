import random
import shutil

from src.configs.config import (
    RAW_DATA_DIR,
    PROCESSED_DIR,
    BASELINE_IMAGE_FILE,
    TUMOR_MASK_PATTERN,
    IMAGE_SIZE,
    HU_MIN,
    HU_MAX,
    SKIP_EMPTY_MASK,
    RESET_PROCESSED_DIR,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)

from src.data.dataset_inspection import get_case_dirs, get_study_dirs
from src.data.preprocess import preprocess_single_study
from src.utils.file_utils import ensure_dir

def collect_valid_studies() -> list[dict]:
    valid_studies = []

    case_dirs = get_case_dirs(RAW_DATA_DIR)

    for case_dir in case_dirs:
        study_dirs = get_study_dirs(case_dir)

        for study_dir in study_dirs:
            image_path = study_dir / BASELINE_IMAGE_FILE
            mask_paths = sorted(study_dir.glob(TUMOR_MASK_PATTERN))

            if image_path.exists() and len(mask_paths) > 0:
                valid_studies.append({
                    "case_id": case_dir.name,
                    "study_id": study_dir.name,
                    "image_path": image_path,
                    "mask_paths": mask_paths,
                })

    return valid_studies

def split_studies(studies: list[dict]) -> dict[str, list[dict]]:
    ratio_sum = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

    studies = list(studies)

    random.seed(RANDOM_SEED)
    random.shuffle(studies)

    total = len(studies)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_studies = studies[:train_end]
    val_studies = studies[train_end:val_end]
    test_studies = studies[val_end:]

    return {
        "train": train_studies,
        "val": val_studies,
        "test": test_studies,
    }

def prepare_processed_dir() -> None:
    if RESET_PROCESSED_DIR and PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    for split_name in ["train", "val", "test"]:
        ensure_dir(PROCESSED_DIR / split_name / "images")
        ensure_dir(PROCESSED_DIR / split_name / "masks")

def main() -> None:
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    print(f"Raw data dir     : {RAW_DATA_DIR}")
    print(f"Processed dir    : {PROCESSED_DIR}")
    print(f"Image file       : {BASELINE_IMAGE_FILE}")
    print(f"Mask pattern     : {TUMOR_MASK_PATTERN}")
    print(f"Image size       : {IMAGE_SIZE}")
    print(f"HU window        : [{HU_MIN}, {HU_MAX}]")
    print(f"Skip empty mask  : {SKIP_EMPTY_MASK}")
    print(f"Reset processed  : {RESET_PROCESSED_DIR}")
    print()

    studies = collect_valid_studies()

    print(f"Valid studies found: {len(studies)}")

    if len(studies) == 0:
        raise RuntimeError("No valid studies found. Check image/mask filenames.")

    split_map = split_studies(studies)

    print(f"Train studies: {len(split_map['train'])}")
    print(f"Val studies  : {len(split_map['val'])}")
    print(f"Test studies : {len(split_map['test'])}")
    print()

    prepare_processed_dir()

    total_saved_all = 0
    failed_studies = []

    for split_name, split_studies_list in split_map.items():
        split_saved = 0

        output_image_dir = PROCESSED_DIR / split_name / "images"
        output_mask_dir = PROCESSED_DIR / split_name / "masks"

        print("=" * 70)
        print(f"Processing split: {split_name}")
        print("=" * 70)

        for item in split_studies_list:
            print(f"Processing: {item['case_id']} / {item['study_id']}")

            try:
                saved_slices = preprocess_single_study(
                    case_id=item["case_id"],
                    study_id=item["study_id"],
                    image_path=item["image_path"],
                    mask_paths=item["mask_paths"],
                    output_image_dir=output_image_dir,
                    output_mask_dir=output_mask_dir,
                    image_size=IMAGE_SIZE,
                    hu_min=HU_MIN,
                    hu_max=HU_MAX,
                    skip_empty_mask=SKIP_EMPTY_MASK,
                )

                split_saved += saved_slices
                total_saved_all += saved_slices

                print(f"Saved slices: {saved_slices}")
                print()

            except Exception as error:
                failed_studies.append({
                    "case_id": item["case_id"],
                    "study_id": item["study_id"],
                    "error": str(error),
                })

                print(f"Failed: {error}")
                print()

        print(f"Total saved for {split_name}: {split_saved}")
        print()

    print("=" * 70)
    print("PREPROCESS SUMMARY")
    print("=" * 70)
    print(f"Total saved slices: {total_saved_all}")
    print(f"Failed studies    : {len(failed_studies)}")
    print(f"Processed dataset : {PROCESSED_DIR}")

    if failed_studies:
        print()
        print("Failed study list:")
        
        for item in failed_studies:
            print(f"- {item['case_id']} / {item['study_id']}: {item['error']}")

    print()
    print("Done.")

if __name__ == "__main__":
    main()