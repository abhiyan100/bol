"""Where the keyword model lives, and how it gets there.

One 17 MB download from the k2-fsa release assets, of which Bol keeps about
5 MB: the int8 encoder, decoder, and joiner, the token table, and the BPE
model that turns a wake phrase into the tokens the decoder can emit. The
float32 copies in the same tarball are another 13 MB that Bol never loads.

Nothing here imports sherpa-onnx, so `bol doctor` can report on the model on
a machine that never installed the wake extra.
"""

from __future__ import annotations

import logging
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from ..config import CONFIG_DIR

log = logging.getLogger("bol.wake")

MODEL_NAME = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    f"{MODEL_NAME}.tar.bz2"
)
# Measured on 2026-09-01 from the release asset listing and the extracted
# files, so `bol setup` can print a disk budget before it spends the bytes.
DOWNLOAD_BYTES = 17_626_723
DISK_BYTES = 5_320_000

_STEM = "epoch-12-avg-2-chunk-16-left-64"

# role -> file name inside the tarball. Flattened on extraction: Bol writes
# the five files it wants by base name, so no archive member can ever choose
# a path outside the model directory.
WANTED = {
    "tokens": "tokens.txt",
    "bpe": "bpe.model",
    "encoder": f"encoder-{_STEM}.int8.onnx",
    "decoder": f"decoder-{_STEM}.int8.onnx",
    "joiner": f"joiner-{_STEM}.int8.onnx",
}


def model_dir() -> Path:
    """Bol's own cache, not the Hugging Face one: this model is a tarball
    from a GitHub release, and it has no repo id to look up."""
    return CONFIG_DIR / "models" / "kws"


def model_files(root: Path | None = None) -> dict[str, Path]:
    """role -> path, whether or not the file is actually there yet."""
    base = Path(root) if root is not None else model_dir()
    return {role: base / name for role, name in WANTED.items()}


def missing_files(root: Path | None = None) -> list[str]:
    return [path.name for path in model_files(root).values() if not path.exists()]


def model_present(root: Path | None = None) -> bool:
    return not missing_files(root)


def human_size(size: float) -> str:
    return f"{size / 1_000_000:.1f} MB"


def download_model(
    root: Path | None = None,
    url: str = MODEL_URL,
    fetch=urllib.request.urlretrieve,
) -> Path:
    """Fetch the tarball and keep the five files Bol loads. Returns the dir.

    Raises whatever the download or the archive raises; the callers turn that
    into one printed line, because a missing wake model costs the wake phrase
    and nothing else.
    """
    base = Path(root) if root is not None else model_dir()
    base.mkdir(parents=True, exist_ok=True)
    wanted = set(WANTED.values())
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "kws.tar.bz2"
        fetch(url, archive)
        with tarfile.open(archive, "r:bz2") as tar:
            for member in tar.getmembers():
                name = Path(member.name).name
                if not member.isfile() or name not in wanted:
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                (base / name).write_bytes(handle.read())
    left = missing_files(base)
    if left:
        raise RuntimeError(
            f"{MODEL_NAME} did not contain {', '.join(left)}. "
            "Delete the folder and run `bol setup` again."
        )
    return base
