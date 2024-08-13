from dataclasses import dataclass

from hipscat.io import FilePointer
from hipscat_import.runtime_arguments import RuntimeArguments

from corrgi.correlation.correlation import Correlation


@dataclass
class CorrgiArguments(RuntimeArguments):
    """Container for Corrgi arguments"""

    left_catalog_path: FilePointer = ""
    """the path to the left catalog"""

    right_catalog_path: FilePointer = ""
    """the path to the cross catalog"""

    correlation: Correlation | None = None
    """correlation instance, with wrappers for each counting method"""

    delete_resume_log_files: bool = False
    """should we delete task-level done files once each stage is complete?
    if False, we will keep all done marker files at the end of the pipeline."""
