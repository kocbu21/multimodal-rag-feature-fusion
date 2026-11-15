"""Make the project root importable when scripts are run as `python scripts/xyz.py`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _cap_vram():
    import os

    gb = os.environ.get("MMRAG_VRAM_CAP_GB")
    if gb:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            torch.cuda.set_per_process_memory_fraction(min(1.0, float(gb) / total))


_cap_vram()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
