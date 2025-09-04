import numpy as np
from distributed import Client
from treecorr import Corr2

from corrgi.pipeline.arguments import CorrgiArguments
from corrgi.pipeline.run_counting import run_counting
import tempfile


def compute_autocorrelation(
    catalog_path: str,
    corr: Corr2,
    client: Client,
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        return run_counting(
            CorrgiArguments(
                left_catalog_path=catalog_path,
                right_catalog_path=catalog_path,
                correlation=corr,
                output_path=tmpdir,
                output_artifact_name="auto_corr",
            ),
            client,
        )


def compute_crosscorrelation(
    left_catalog_path: str,
    right_catalog_path: str,
    corr: Corr2,
    client: Client,
) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        return run_counting(
            CorrgiArguments(
                left_catalog_path=left_catalog_path,
                right_catalog_path=right_catalog_path,
                correlation=corr,
                output_path=tmpdir,
                output_artifact_name="cross_corr",
            ),
            client,
        )
