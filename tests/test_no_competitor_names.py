"""Bol describes its own behavior; competitor names stay out of the tree."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANNED = ("infina",)
SCAN = ("bol", "tests", "docs", "site", "README.md", "pyproject.toml")
SUFFIXES = {".py", ".md", ".toml", ".html", ".txt", ".json"}


def _files():
    for entry in SCAN:
        path = ROOT / entry
        if path.is_file():
            yield path
        elif path.is_dir():
            for sub in path.rglob("*"):
                if sub.is_file() and sub.suffix in SUFFIXES and "__pycache__" not in sub.parts:
                    yield sub


def test_no_competitor_names_anywhere():
    hits = []
    for path in _files():
        if path.name == "test_no_competitor_names.py":
            continue
        text = path.read_text(errors="ignore").lower()
        for word in BANNED:
            if word in text:
                hits.append(f"{path.relative_to(ROOT)}: {word}")
    assert hits == []
