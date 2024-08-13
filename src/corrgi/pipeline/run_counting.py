"""Compute correlation using dask for parallelization

Methods in this file set up a dask pipeline using futures.
The actual logic of the map reduce is in the `map_reduce.py` file."""

import numpy as np
from dask.distributed import Client
from hipscat.io import paths

import corrgi.pipeline.map_reduce as mr
from corrgi.pipeline.arguments import CorrgiArguments
from corrgi.pipeline.resume_plan import CorrgiResumePlan


def run_pipeline(args: CorrgiArguments):
    """Pipeline that creates its own client from the provided runtime arguments"""
    with Client(
        local_directory=args.dask_tmp,
        n_workers=args.dask_n_workers,
        threads_per_worker=args.dask_threads_per_worker,
    ) as client:
        return run_counting(args, client)


def run_counting(args, client):
    """Run counting of pairs in a map-reduce pipeline.

    The pipeline sends a task to each worker for each left partition of the alignment
    to perform counts on all the corresponding right catalog partitions. This reduces
    the amount of reads and therefore the I/O overhead."""
    resume_plan = CorrgiResumePlan(args)

    # Compute the partial histograms for auto-correlation
    if not resume_plan.is_mapping_auto_done():
        auto_futures = get_autocorrelation_futures(args, resume_plan, client)
        resume_plan.wait_for_auto_mapping(auto_futures)

    # Compute the partial histograms for cross-correlation
    if not resume_plan.is_mapping_cross_done():
        cross_futures = get_crosscorrelation_futures(args, resume_plan, client)
        resume_plan.wait_for_cross_mapping(cross_futures)

    # Reduce all intermediate histograms to pixel-histograms
    if not resume_plan.is_reducing_done():
        reducing_future = get_reducing_future(args, resume_plan, client)
        resume_plan.wait_for_reducing(reducing_future)

    # All done - cleaning up intermediate files
    resume_plan.clean_resume_files()

    # Read the final histogram and return for further processing
    return np.load(resume_plan.output_artifact_path)


def get_autocorrelation_futures(args, resume_plan, client):
    auto_futures = []
    for pixel, mapping_key in resume_plan.get_remaining_map_auto_keys().items():
        partition_file = paths.pixel_catalog_file(args.left_catalog_path, pixel.order, pixel.pixel)
        auto_futures.append(
            client.submit(
                mr.map_pixel_auto_counts,
                partition_file=partition_file,
                catalog_info=args.left_hc_catalog.catalog_info,
                correlation=args.correlation,
                mapping_key=mapping_key,
                resume_path=resume_plan.tmp_path,
            )
        )
    return auto_futures


def get_crosscorrelation_futures(args, resume_plan, client):
    cross_futures = []
    for left_pixel, (right_pixels, mapping_keys) in resume_plan.get_remaining_map_cross_keys().items():
        left_partition_file = paths.pixel_catalog_file(
            args.left_catalog_path, left_pixel.order, left_pixel.pixel
        )
        right_partition_files = paths.pixel_catalog_files(args.right_catalog_path, right_pixels)
        cross_futures.append(
            client.submit(
                mr.map_pixel_cross_counts,
                left_partition_file=left_partition_file,
                right_partition_files=right_partition_files,
                left_catalog_info=args.left_hc_catalog.catalog_info,
                right_catalog_info=args.right_hc_catalog.catalog_info,
                correlation=args.correlation,
                mapping_keys=mapping_keys,
                resume_path=resume_plan.tmp_path,
            )
        )
    return cross_futures


def get_reducing_future(args, resume_plan, client):
    return client.submit(
        mr.reduce_pixel_counts,
        reducing_keys=resume_plan.get_reducing_keys(),
        resume_path=resume_plan.tmp_path,
        output_artifact_path=resume_plan.output_artifact_path,
        delete_resume_log_files=args.delete_resume_log_files,
    )
