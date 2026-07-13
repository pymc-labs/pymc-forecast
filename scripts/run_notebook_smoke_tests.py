"""Execute every example notebook with deliberately tiny inference schedules."""

from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import nbformat
from ipykernel.kernelspec import install as install_kernel
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "docs" / "examples"
KERNEL_NAME = "pymc-forecast-smoke"

REPLACEMENTS = (
    (r"(?m)^(\s*[A-Z_]*SAMPLES?\s*=\s*)[\d_]+", r"\g<1>20"),
    (r"(?m)^(\s*num_samples\s*=\s*)[\d_]+", r"\g<1>20"),
    (r"\bnum_samples\s*=\s*[\d_]+", "num_samples=20"),
    (r"\bnum_steps\s*=\s*[\d_]+", "num_steps=50"),
    (r"\bdraws\s*=\s*[\d_]+", "draws=20"),
    (r"\btune\s*=\s*[\d_]+", "tune=20"),
    (r"\bchains\s*=\s*[\d_]+", "chains=1"),
    (r'"draws"\s*:\s*[\d_]+', '"draws": 20'),
    (r'"tune"\s*:\s*[\d_]+', '"tune": 20'),
    (r'"chains"\s*:\s*[\d_]+', '"chains": 1'),
    (r"min_train_window=104,", "min_train_window=duration - 52,"),
    (r"stride=52,", "stride=duration,"),
)


def smoke_source(source: str) -> str:
    """Return notebook source with only inference budgets reduced."""
    for pattern, replacement in REPLACEMENTS:
        source = re.sub(pattern, replacement, source)
    return source


def execute_notebook(path: Path) -> None:
    """Execute one notebook in a temporary directory without changing it."""
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.source = smoke_source(cell.source)
    with tempfile.TemporaryDirectory(prefix=f"{path.stem}-") as workdir:
        client = NotebookClient(
            notebook,
            timeout=1_200,
            kernel_name=KERNEL_NAME,
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute(cwd=workdir)


def main() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    notebooks = [Path(item).resolve() for item in sys.argv[1:]] or sorted(EXAMPLES.glob("*.ipynb"))
    if not notebooks:
        raise RuntimeError(f"no notebooks found in {EXAMPLES}")
    with tempfile.TemporaryDirectory(prefix="pymc-forecast-kernel-") as prefix:
        install_kernel(prefix=prefix, kernel_name=KERNEL_NAME, display_name=KERNEL_NAME)
        kernel_path = str(Path(prefix) / "share" / "jupyter")
        existing_path = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = os.pathsep.join(filter(None, [kernel_path, existing_path]))
        for notebook in notebooks:
            print(f"smoke testing {notebook.relative_to(ROOT)}", flush=True)
            execute_notebook(notebook)


if __name__ == "__main__":
    main()
