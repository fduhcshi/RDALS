"""
Entry point for reweighting visualization.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .reweight_kde import main

if __name__ == '__main__':
    main()
