from __future__ import annotations

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
ARCHIVE = DIST / "qpeak_neurips_code.zip"

EXCLUDE_DIRS = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    ".yoyo",
    "Code",
    "__pycache__",
    "_runs",
    "dist",
    "env",
    "venv",
}

EXCLUDE_FILE_NAMES = {
    ".DS_Store",
}

EXCLUDE_SUFFIXES = {
    ".dll",
    ".dylib",
    ".pyc",
    ".pyo",
    ".so",
}


def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return False
    if path.name in EXCLUDE_FILE_NAMES:
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    if path.is_dir() and path.name.endswith("_figs"):
        return False
    return True


def main() -> None:
    DIST.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        ARCHIVE.unlink()

    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and should_include(p))
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(ROOT))

    print(f"Wrote {ARCHIVE}")
    print(f"Included {len(files)} files")


if __name__ == "__main__":
    main()
