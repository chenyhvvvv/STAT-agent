"""
Notebook Logger for STAT Agent.

Logs all code execution during a conversation session as a single Jupyter notebook.
Uses read-modify-write so user edits in Jupyter are preserved.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import logging

logger = logging.getLogger(__name__)

# Notebook skeleton
NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.10.0",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "file_extension": ".py"
    }
}


def _make_source(code: str) -> List[str]:
    """Convert code string to Jupyter source format (each line ends with \\n except last)."""
    lines = code.split("\n")
    if len(lines) <= 1:
        return lines
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def _make_stream(text: str, name: str = "stdout") -> Dict:
    """Create a Jupyter stream output."""
    return {
        "output_type": "stream",
        "name": name,
        "text": [line + "\n" for line in text.split("\n")]
    }


class NotebookLogger:
    """
    Logger for capturing code execution in Jupyter notebook format.

    Uses read-modify-write: on each save, reads the notebook from disk,
    appends pending cells, and writes back. This preserves any edits
    the user made in Jupyter.
    """

    def __init__(self, log_dir: str = "logs", enabled: bool = True):
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.session_dir: Optional[Path] = None
        self.notebook_file: Optional[Path] = None
        self.session_start_time: Optional[datetime] = None
        self.initialized: bool = False

        # Pending cells not yet written to disk
        self._pending_cells: List[Dict] = []

        if self.enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def initialize_notebook(self, dataset_path: str, llm_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize notebook when user loads a dataset."""
        if not self.enabled or self.initialized:
            return

        self.session_start_time = datetime.now()
        timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / f"session_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.notebook_file = self.session_dir / "analysis.ipynb"

        logger.info(f"NotebookLogger: session={self.session_dir}, file={self.notebook_file}")

        # Build initialization cell
        init_code = [
            "# STAT Agent Analysis Session",
            f"# Started: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "# Import required libraries",
            "from stat_agent.core.session import SimpleSession",
            "import numpy as np",
            "import pandas as pd",
            "import scanpy as sc",
            "import matplotlib.pyplot as plt",
            "",
            "# Initialize session",
            "session = SimpleSession()",
            "",
            "# Load dataset",
            f"session.load_dataset('{dataset_path}')",
            ""
        ]

        # LLM configuration
        if llm_config:
            init_code.extend([
                "# LLM configuration (from web interface session)",
                "session.llm_config = {",
                f"    'api_key': '{llm_config.get('api_key', 'YOUR_API_KEY')}',",
                f"    'model': '{llm_config.get('model', 'claude-sonnet-3-5-20241022')}',",
                f"    'base_url': {repr(llm_config.get('base_url'))}",
                "}",
                ""
            ])
        else:
            init_code.extend([
                "# LLM configuration (set this for LLM-based skills)",
                "# session.llm_config = {",
                "#     'api_key': 'YOUR_API_KEY',",
                "#     'model': 'claude-sonnet-3-5-20241022',",
                "#     'base_url': None",
                "# }",
                ""
            ])

        init_code.extend([
            "# Quick access",
            "slice_0 = session.get_slice(0)",
            "adata = slice_0.adata if slice_0 else None",
            "",
            "print(f'Loaded {len(session.slices)} slice(s)')",
            "if adata is not None:",
            "    print(f'Slice 0: {adata.n_obs} cells, {adata.n_vars} genes')"
        ])

        self._append_cell("\n".join(init_code))

        # Skills setup cell
        skills_code = [
            "# Setup: Load all STAT skills",
            "import sys, os",
            "from pathlib import Path",
            "",
            "try:",
            "    import stat_agent",
            "    SKILLS_DIR = Path(stat_agent.__file__).parent.parent / '.claude' / 'skills'",
            "except ImportError:",
            "    SKILLS_DIR = Path(os.environ.get('STAT_SKILLS_DIR', './skills'))",
            "",
            "if SKILLS_DIR.exists():",
            "    for p in SKILLS_DIR.iterdir():",
            "        if p.is_dir() and not p.name.startswith('.'):",
            "            sys.path.insert(0, str(p))",
            "    print(f'Loaded {len([p for p in SKILLS_DIR.iterdir() if p.is_dir()])} skills')",
            "else:",
            "    print(f'Skills directory not found: {SKILLS_DIR}')"
        ]

        self._append_cell("\n".join(skills_code))

        self.initialized = True
        self.save()
        logger.info(f"Notebook initialized: {self.notebook_file}")

    def append_roi_creation(self, roi_name: str, slice_id: int, roi_definition: Dict[str, Any]) -> None:
        """Append ROI creation code when user draws ROI in canvas."""
        if not self.enabled or not self.initialized:
            return

        code_lines = [
            f"# ROI created: {roi_name} (slice {slice_id})",
            f"session.create_roi('{roi_name}', {slice_id}, {repr(roi_definition)})",
            f"roi = session.get_roi('{roi_name}')",
            f"print(f'ROI {{roi.name}} on slice {{roi.slice_id}}: {{roi.n_obs}} cells')"
        ]

        self._append_cell("\n".join(code_lines))
        self.save()

    def append_code_execution(
        self,
        code: str,
        result: Any,
        user_query: Optional[str] = None,
        context: Optional[str] = None,
        execution_time: Optional[float] = None
    ) -> None:
        """Append code execution with results."""
        if not self.enabled or not self.initialized:
            return

        # Build code with context comments
        header = []
        if user_query:
            header.append(f"# User: {user_query}")
        if context:
            header.append(f"# {context}")
        if header:
            header.append("")
        full_code = "\n".join(header) + code if header else code

        # Build outputs
        outputs = []
        if result and result.stdout and result.stdout.strip():
            outputs.append(_make_stream(result.stdout, "stdout"))
        if result and result.stderr and result.stderr.strip():
            outputs.append(_make_stream(result.stderr, "stderr"))
        if result and result.plots:
            for plot_base64 in result.plots:
                outputs.append({
                    "output_type": "display_data",
                    "data": {"image/png": plot_base64},
                    "metadata": {}
                })
        if execution_time is not None:
            timing = f"Execution time: {execution_time:.2f}s"
            if outputs and outputs[-1].get("name") == "stdout":
                outputs[-1]["text"].append(f"\n{timing}")
            else:
                outputs.append(_make_stream(timing, "stdout"))

        self._append_cell(full_code, outputs)
        self.save()

    def _append_cell(self, code: str, outputs: Optional[List[Dict]] = None) -> None:
        """Add a code cell to the pending buffer."""
        self._pending_cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": _make_source(code),
            "outputs": outputs or []
        })

    def _read_notebook(self) -> Dict:
        """Read notebook from disk, or return empty skeleton."""
        if self.notebook_file and self.notebook_file.exists():
            try:
                with open(self.notebook_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read notebook, starting fresh: {e}")
        return {
            "cells": [],
            "metadata": {**NOTEBOOK_METADATA},
            "nbformat": 4,
            "nbformat_minor": 5
        }

    def save(self) -> None:
        """Read-modify-write: read from disk, append pending cells, write back."""
        if not self.enabled or not self.initialized or not self.notebook_file:
            return
        if not self._pending_cells:
            return

        notebook = self._read_notebook()
        notebook["cells"].extend(self._pending_cells)
        self._pending_cells.clear()

        try:
            with open(self.notebook_file, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved notebook: {self.notebook_file} ({len(notebook['cells'])} cells)")
        except Exception as e:
            logger.error(f"Failed to save notebook: {e}")

    def disable(self) -> None:
        self.enabled = False

    def enable(self) -> None:
        self.enabled = True
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_session_dir(self) -> Optional[Path]:
        return self.session_dir

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        init = "initialized" if self.initialized else "not initialized"
        pending = len(self._pending_cells)
        return f"NotebookLogger({status}, {init}, pending={pending})"


__all__ = ['NotebookLogger']
