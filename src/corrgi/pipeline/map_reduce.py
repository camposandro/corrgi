from typing import List

import numpy as np
from hipscat.catalog.catalog_info import CatalogInfo
from hipscat.io import FilePointer, file_io
from hipscat_import.pipeline_resume_plan import print_task_failure

from corrgi.correlation.correlation import Correlation
from corrgi.pipeline.resume_plan import CorrgiResumePlan


def map_pixel_auto_counts(
    partition_file: FilePointer,
    catalog_info: CatalogInfo,
    correlation: Correlation,
    mapping_key: str,
    resume_path: FilePointer,
):
    try:
        left_df = file_io.read_parquet_file_to_pandas(partition_file)
        hist = correlation.count_auto_pairs(left_df, catalog_info)
        filename = CorrgiResumePlan.get_histogram_filepath(tmp_path=resume_path, mapping_key=mapping_key)
        np.save(filename, hist)
        CorrgiResumePlan.mapping_key_done(tmp_path=resume_path, mapping_key=mapping_key)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed MAPPING auto stage for file {partition_file}", exception)
        raise exception


def map_pixel_cross_counts(
    left_partition_file: FilePointer,
    right_partition_files: List[FilePointer],
    left_catalog_info: CatalogInfo,
    right_catalog_info: CatalogInfo,
    correlation: Correlation,
    mapping_keys: List[str],
    resume_path: FilePointer,
):
    try:
        left_df = file_io.read_parquet_file_to_pandas(left_partition_file)
        for right_partition, mapping_key in zip(right_partition_files, mapping_keys):
            right_df = file_io.read_parquet_file_to_pandas(right_partition)
            hist = correlation.count_cross_pairs(left_df, right_df, left_catalog_info, right_catalog_info)
            filename = CorrgiResumePlan.get_histogram_filepath(tmp_path=resume_path, mapping_key=mapping_key)
            np.save(filename, hist)
            CorrgiResumePlan.mapping_key_done(tmp_path=resume_path, mapping_key=mapping_key)
            del right_df
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed cross MAPPING stage for file {left_partition_file}", exception)
        raise exception


def reduce_pixel_counts(
    reducing_keys: List[str],
    resume_path: FilePointer,
    output_artifact_path: str,
    delete_resume_log_files: bool,
):
    try:
        histogram = None
        for path in reducing_keys:
            partial_histogram = np.load(path)
            histogram = histogram + partial_histogram if histogram is not None else partial_histogram
            del partial_histogram
        np.save(output_artifact_path, histogram)
        if delete_resume_log_files:
            file_io.remove_directory(resume_path, ignore_errors=True)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        print_task_failure(f"Failed REDUCING stage", exception)
        raise exception
