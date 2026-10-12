"""Fresh, exploratory Track C pilot. No old trajectories or checkpoints."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.pilot_deps'))
# Load NumPy first to avoid conflicting OpenMP initialization in this host's Conda.
import numpy as np
