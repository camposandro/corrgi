from __future__ import annotations

import numpy as np
from hipscat.io import FilePointer

from corrgi.pipeline.arguments import CorrgiArguments
from corrgi.pipeline.run_counting import run_pipeline
from corrgi.estimators.estimator import Estimator


class DavisPeeblesEstimator(Estimator):
    """Davis-Peebles Estimator"""

    def compute_autocorrelation_counts(
        self, catalog_path: FilePointer, random_catalog_path: FilePointer
    ) -> list[np.ndarray, np.ndarray, np.ndarray | int]:
        """Computes the auto-correlation counts for the provided catalog"""
        raise NotImplementedError()

    def compute_crosscorrelation_counts(
        self,
        left_catalog_path: FilePointer,
        right_catalog_path: FilePointer,
        random_catalog_path: FilePointer,
    ) -> list[np.ndarray, np.ndarray]:
        """Computes the cross-correlation counts for the provided catalog.

        Args:
            left_catalog_path (str): A left galaxy samples catalog (D).
            right_catalog_path (str): A right galaxy samples catalog (C).
            random_catalog_path (str): A random samples catalog (R).

        Returns:
            The CD and CR counts for the DP estimator.
        """
        counts_cd = run_pipeline(
            CorrgiArguments(
                left_catalog_path=right_catalog_path,
                right_catalog_path=left_catalog_path,
                correlation=self.correlation,
                output_path=self.output_dir,
                output_artifact_name="cd",
                simple_progress_bar=True,
            )
        )
        counts_cr = run_pipeline(
            CorrgiArguments(
                left_catalog_path=right_catalog_path,
                right_catalog_path=random_catalog_path,
                correlation=self.correlation,
                output_path=self.output_dir,
                output_artifact_name="cr",
                simple_progress_bar=True,
            )
        )
        return self.correlation.transform_counts([counts_cd, counts_cr])
