from csv import DictWriter
from datetime import datetime
from pathlib import Path

def make_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: Path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def build_timestamped_path(output_dir: Path, prefix: str, suffix: str):
    output_dir = ensure_dir(output_dir)
    timestamp = make_timestamp()
    return output_dir / f"{prefix}_{timestamp}{suffix}"

def write_rows_to_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_lines_to_file(path: Path, lines: list[str]):
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))