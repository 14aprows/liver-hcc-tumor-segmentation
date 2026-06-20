from src.configs.config import (
    RAW_DATA_DIR,
    IMAGE_FILES,
    LIVER_MASK_FILES,
    DATASET_CHECK_LOGS_DIR,
)
from src.data.dataset_inspection import get_case_dirs, get_study_dirs, file_exists
from src.utils.nifti_utils import load_nifti_shape, read_mask_info
from src.utils.file_utils import (
    build_timestamped_path,
    write_rows_to_csv,
    # write_lines_to_file,
)

def check_dataset():
    case_dirs = get_case_dirs(RAW_DATA_DIR)

    total_cases = 0
    total_studies = 0
    missing_files = []
    shape_errors = []

    rows = []
    # log_lines = []

    csv_path = build_timestamped_path(
        DATASET_CHECK_LOGS_DIR,
        prefix="dataset_check",
        suffix=".csv",
    )

    # log_path = build_timestamped_path(
    #     DATASET_CHECK_LOGS_DIR,
    #     prefix="dataset_check",
    #     suffix=".log",
    # )

    print(f"Raw data dir: {RAW_DATA_DIR}")
    print(f"Total cases: {len(case_dirs)}")
    print()

    for case_dir in case_dirs:
        study_dirs = get_study_dirs(case_dir)
        total_cases += 1
        total_studies += len(study_dirs)

        for study_dir in study_dirs:
            print(f"Checking: {case_dir.name} / {study_dir.name}")

            required_files = IMAGE_FILES + LIVER_MASK_FILES

            study_missing_files = [
                filename for filename in required_files
                if not file_exists(study_dir, filename)
            ]

            missing_files.extend(study_dir / filename for filename in study_missing_files)

            existing_files = [
                filename for filename in required_files
                if file_exists(study_dir, filename)
            ]

            shapes = {
                filename: load_nifti_shape(study_dir / filename)
                for filename in existing_files
            }

            unique_shapes = set(shapes.values())
            has_shape_mismatch = len(unique_shapes) > 1

            if has_shape_mismatch:
                shape_errors.append((study_dir, shapes))

            tumor_masks = sorted(study_dir.glob("rater1_tumor*.nii.gz"))

            for filename in required_files:
                file_path = study_dir / filename
                mask_info = {}

                if filename in LIVER_MASK_FILES and file_path.exists():
                    mask_info = read_mask_info(file_path)
                    print(f"  {filename}: {mask_info}")

                rows.append({
                    "case_id": case_dir.name,
                    "study_date": study_dir.name,
                    "filename": filename,
                    "exists": file_path.exists(),
                    "shape": shapes.get(filename, ""),
                    "is_missing": filename in study_missing_files,
                    "has_shape_mismatch": has_shape_mismatch,
                    "mask_unique_values": mask_info.get("unique_values", "") if isinstance(mask_info, dict) else "",
                    "positive_voxels": mask_info.get("positive_voxels", "") if isinstance(mask_info, dict) else "",
                    "rater1_tumor_mask_count": len(tumor_masks),
                    "study_dir": str(study_dir),
                })

            # log_lines.append(
            #     f"{case_dir.name}/{study_dir.name} | "
            #     f"missing={len(study_missing_files)} | "
            #     f"shape_mismatch={has_shape_mismatch} | "
            #     f"rater1_tumor_masks={len(tumor_masks)}"
            # )

            print(f"rater1 tumor masks: {len(tumor_masks)}")
            print()

    fieldnames = [
        "case_id",
        "study_date",
        "filename",
        "exists",
        "shape",
        "is_missing",
        "has_shape_mismatch",
        "mask_unique_values",
        "positive_voxels",
        "rater1_tumor_mask_count",
        "study_dir",
    ]

    write_rows_to_csv(csv_path, rows, fieldnames)
    # write_lines_to_file(log_path, log_lines)

    print("Summary")
    print(f"Total cases: {total_cases}")
    print(f"Total studies: {total_studies}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Shape errors: {len(shape_errors)}")
    print(f"CSV saved to: {csv_path}")
    # print(f"Log saved to: {log_path}")

if __name__ == "__main__":
    check_dataset()