from __future__ import annotations

from dataclasses import dataclass

import lsdb
from hats_import.runtime_arguments import RuntimeArguments
from treecorr import Corr2


@dataclass
class CorrgiArguments(RuntimeArguments):
    """Container for Corrgi arguments"""

    left_catalog: lsdb.Catalog | None = None
    """the left catalog"""

    right_catalog: lsdb.Catalog | None = None
    """the right catalog"""

    correlation: Corr2 | None = None
    """correlation instance, with wrappers for each counting method"""

    simple_progress_bar: bool = True
    """use plain-text progress bar"""

    corr_type: type[Corr2] | None = None
    """the TreeCorr correlation type to use"""

    corr_args: dict | None = None
    """arguments to pass to the correlation constructor"""

    def __post_init__(self):
        self._check_arguments()
        self.left_catalog_path = self.left_catalog.hc_structure.catalog_path
        self.right_catalog_path = self.right_catalog.hc_structure.catalog_path
