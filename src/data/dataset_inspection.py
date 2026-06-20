from pathlib import Path

def get_case_dirs(raw_data_dir: Path):
    raw_data_dir = Path(raw_data_dir)    
    return sorted(
        path for path in raw_data_dir.iterdir()
        if path.is_dir() and path.name.startswith("TCGA-")
    )

def get_study_dirs(case_dir: Path):
    case_dir = Path(case_dir)
    return sorted(
        path for path in case_dir.iterdir()
        if path.is_dir()
    )

def get_nifti_files(study_dir: Path):
    study_dir = Path(study_dir)
    return sorted(study_dir.glob("*.nii.gz"))

def file_exists(study_dir: Path, filename: str):
    study_dir = Path(study_dir)
    return (study_dir / filename).exists()


