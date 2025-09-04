import numpy as np
import pandas as pd
import treecorr
from hats.io import file_io
from hats_import.pipeline_resume_plan import print_task_failure

from corrgi.pipeline.resume_plan import CorrgiResumePlan
from treecorr import Corr2


def map_pixel_auto_counts(
    partition_file: str,
    ra_column: str,
    dec_column: str,
    correlation: Corr2,
    mapping_key: str,
    resume_path: str,
):
    """Computes counts in partitions for points within themselves"""
    try:
        df = pd.read_parquet(partition_file, dtype_backend="pyarrow", memory_map=True)
        # Compute auto-pairs using TreeCorr
        cat = treecorr.Catalog(
            ra=df[ra_column].values,
            dec=df[dec_column].values,
            ra_units="deg",
            dec_units="deg",
            # TODO: May need to add more params if using GGCorrelation etc.
        )
        correlation.process_auto(cat)
        # Save histogram as a npy
        filename = CorrgiResumePlan.get_histogram_filepath(tmp_path=resume_path, mapping_key=mapping_key)
        hist = correlation.npairs
        np.save(filename, hist)
        CorrgiResumePlan.mapping_key_done(tmp_path=resume_path, mapping_key=mapping_key)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed MAPPING auto stage for file {partition_file}", exception)
        raise exception


def map_pixel_cross_counts(
    left_partition_file: str,
    right_partition_files: list[str],
    left_ra_column: str,
    left_dec_column: str,
    right_ra_column: str,
    right_dec_column: str,
    correlation: Corr2,
    mapping_keys: list[str],
    resume_path: str,
):
    """Computes counts for points in different partitions"""
    try:
        left_df = file_io.read_parquet_file_to_pandas(left_partition_file)
        for right_partition, mapping_key in zip(right_partition_files, mapping_keys):
            right_df = file_io.read_parquet_file_to_pandas(right_partition)
            # Compute cross-pairs using TreeCorr
            cat1 = treecorr.Catalog(
                ra=left_df[left_ra_column].values,
                dec=left_df[left_dec_column].values,
                ra_units="deg",
                dec_units="deg",
                # TODO: May need to add more params if using GGCorrelation etc.
            )
            cat2 = treecorr.Catalog(
                ra=right_df[right_ra_column].values,
                dec=right_df[right_dec_column].values,
                ra_units="deg",
                dec_units="deg",
                # TODO: May need to add more params if using GGCorrelation etc.
            )
            correlation.process_cross(cat1, cat2)
            # Save histogram as a npy
            filename = CorrgiResumePlan.get_histogram_filepath(tmp_path=resume_path, mapping_key=mapping_key)
            hist = correlation.npairs
            np.save(filename, hist)
            CorrgiResumePlan.mapping_key_done(tmp_path=resume_path, mapping_key=mapping_key)
            del right_df
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed cross MAPPING stage for file {left_partition_file}", exception)
        raise exception


def reduce_pixel_counts(reducing_keys: list[str], output_artifact_path: str):
    """Sums all the intermediate counts to a final histogram"""
    try:
        histogram = None
        for path in reducing_keys:
            partial_histogram = np.load(path)
            histogram = histogram + partial_histogram if histogram is not None else partial_histogram
            del partial_histogram
        np.save(output_artifact_path, histogram)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure("Failed REDUCING stage", exception)
        raise exception
