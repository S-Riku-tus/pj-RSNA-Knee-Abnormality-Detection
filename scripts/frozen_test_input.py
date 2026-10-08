"""Resolve the two supported Kaggle layouts by live test contents, without MRI reads."""

from pathlib import Path


def resolve_test_input(input_root="/kaggle/input"):
    """Return one complete test mount; a placeholder directory never shadows the data."""
    root = Path(input_root)
    relative = (
        "competitions/rsna-knee-abnormality-detection",
        "rsna-knee-abnormality-detection",
    )
    required_files = ("test.csv", "test_series.csv")
    valid = set()
    for name in relative:
        candidate = root / name
        if (
            all((candidate / filename).is_file() for filename in required_files)
            and (candidate / "test_series").is_dir()
        ):
            valid.add(candidate.resolve())
    if len(valid) != 1:
        raise RuntimeError(
            "Expected one live competition test root containing test.csv, test_series.csv and test_series/; "
            f"found {len(valid)} distinct complete roots in the two supported Kaggle layouts"
        )
    return valid.pop()
