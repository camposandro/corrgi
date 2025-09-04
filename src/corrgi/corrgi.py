from typing import Type

import lsdb
import numpy as np
from distributed import Client
from treecorr import Corr2

from corrgi.pipeline.arguments import CorrgiArguments
from corrgi.pipeline.run_counting import run_counting
import tempfile


def compute_autocorrelation(
    catalog: lsdb.Catalog,
    corr_type: Type[Corr2],
    corr_args: dict,
    client: Client,
) -> np.ndarray:
    """Compute auto-correlation given an LSDB Catalog."""
    with tempfile.TemporaryDirectory() as tmpdir:
        return run_counting(
            CorrgiArguments(
                left_catalog=catalog,
                right_catalog=catalog,
                corr_type=corr_type,
                corr_args=corr_args,
                output_path=tmpdir,
                output_artifact_name="auto",
            ),
            client,
        )


def compute_crosscorrelation(
    left_catalog: lsdb.Catalog,
    right_catalog: lsdb.Catalog,
    corr_type: Type[Corr2],
    corr_args: dict,
    client: Client,
) -> np.ndarray:
    """Compute cross-correlation given two LSDB Catalogs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        return run_counting(
            CorrgiArguments(
                left_catalog=left_catalog,
                right_catalog=right_catalog,
                corr_type=corr_type,
                corr_args=corr_args,
                output_path=tmpdir,
                output_artifact_name="cross",
            ),
            client,
        )
