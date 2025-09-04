from __future__ import annotations

from dataclasses import dataclass

import hats as hc
from hats_import.runtime_arguments import RuntimeArguments
from treecorr import Corr2


@dataclass
class CorrgiArguments(RuntimeArguments):
    """Container for Corrgi arguments"""

    left_catalog_path: str = ""
    """the path to the left catalog"""

    right_catalog_path: str = ""
    """the path to the cross catalog"""

    correlation: Corr2 | None = None
    """correlation instance, with wrappers for each counting method"""

    simple_progress_bar: bool = True
    """use plain-text progress bar"""

    def __post_init__(self):
        self._check_arguments()
        self.left_catalog_path = str(self.left_catalog_path)
        self.right_catalog_path = str(self.right_catalog_path)
        self.left_hc_catalog = hc.read_hats(self.left_catalog_path)
        self.right_hc_catalog = hc.read_hats(self.right_catalog_path)
