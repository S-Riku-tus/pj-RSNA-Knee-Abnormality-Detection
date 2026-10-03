"""Filesystem comparisons shared by export transfer and verification."""

import sys
from pathlib import Path, PureWindowsPath


def comparable_path(path):
    """Compare Windows extended-length and ordinary spellings consistently."""
    text = str(path)
    if text.startswith("\\\\?\\UNC\\"):
        return PureWindowsPath("\\\\" + text[8:])
    if text.startswith("\\\\?\\"):
        return PureWindowsPath(text[4:])
    return PureWindowsPath(text) if sys.platform == "win32" else Path(text)
