import dask
import numpy as np
import pandas as pd
import treecorr
from dask.distributed import print as dask_print
from hats.catalog import TableProperties
from hats.pixel_math import HealpixPixel
from lsdb import Catalog
from lsdb.dask.merge_catalog_functions import (
    align_and_apply,
    get_healpix_pixels_from_alignment,
)

from corrgi.alignment import autocorrelation_alignment, crosscorrelation_alignment
from treecorr import Corr2


def compute_autocorrelation(catalog: Catalog, corr: Corr2) -> np.ndarray:
    """Aligns the pixel of a single catalog and performs the pairs counting.

    Args:
        catalog (Catalog): The catalog.
        corr: The TreeCorr correlation instance.

    Returns:
        The histogram with the sample distance counts.
    """
    # Get counts between points of different partitions
    alignment = autocorrelation_alignment(catalog.hc_structure)
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    cross_partials = align_and_apply(
        [(catalog, left_pixels), (catalog, right_pixels)],
        count_cross_pairs,
        corr,
    )
    # Get counts between points of the same partition
    auto_partials = [
        count_auto_pairs(partition, catalog.hc_structure.catalog_info, corr)
        for partition in catalog._ddf.to_delayed()
    ]
    all_partials = [*cross_partials, *auto_partials]
    return join_count_histograms(all_partials)


def compute_crosscorrelation(left: Catalog, right: Catalog, corr: Corr2) -> np.ndarray:
    """Aligns the pixel of two catalogs and performs the pairs counting.

    Args:
        left (Catalog): The left catalog.
        right (Catalog): The right catalog.
        corr: The TreeCorr correlation instance.

    Returns:
        The histogram with the sample distance counts.
    """
    alignment = crosscorrelation_alignment(left.hc_structure, right.hc_structure)
    left_pixels, right_pixels = get_healpix_pixels_from_alignment(alignment)
    cross_partials = align_and_apply([(left, left_pixels), (right, right_pixels)], count_cross_pairs, corr)
    return join_count_histograms(cross_partials)


@dask.delayed
def count_auto_pairs(
    df: pd.DataFrame,
    catalog_info: TableProperties,
    corr: Corr2,
) -> np.ndarray:
    """Calls the TreeCorr routine to compute the counts for pairs of partitions
    belonging to the same catalog.

    Args:
       df (pd.DataFrame): The partition dataframe.
       catalog_info (TableProperties): The catalog metadata.
       corr (Corr2): The correlation instance.

    Returns:
       The count histogram for the partition pair.
    """
    try:
        cat = treecorr.Catalog(
            ra=df[catalog_info.ra_column].values,
            dec=df[catalog_info.dec_column].values,
            ra_units="deg",
            dec_units="deg",
        )
        corr.process(cat)
        return corr.npairs
    except Exception as exception:
        dask_print(exception)
        raise exception


@dask.delayed
def count_cross_pairs(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_pix: HealpixPixel,
    right_pix: HealpixPixel,
    left_catalog_info: TableProperties,
    right_catalog_info: TableProperties,
    corr: Corr2,
) -> np.ndarray:
    """Calls the fortran routine to compute the counts for pairs of
    partitions belonging to two different catalogs.

    Args:
       left_df (pd.DataFrame): The left partition dataframe.
       right_df (pd.DataFrame): The right partition dataframe.
       left_pix (HealpixPixel): The pixel corresponding to `left_df`.
       right_pix (HealpixPixel): The pixel corresponding to `right_df`.
       left_catalog_info (TableProperties): The left catalog metadata.
       right_catalog_info (TableProperties): The right catalog metadata.
       corr (Corr2): The correlation instance.

    Returns:
       The count histogram for the partition pair.
    """
    try:
        cat1 = treecorr.Catalog(
            ra=left_df[left_catalog_info.ra_column].values,
            dec=left_df[left_catalog_info.dec_column].values,
            ra_units="deg",
            dec_units="deg",
        )
        cat2 = treecorr.Catalog(
            ra=right_df[right_catalog_info.ra_column].values,
            dec=right_df[right_catalog_info.dec_column].values,
            ra_units="deg",
            dec_units="deg",
        )
        corr.process(cat1, cat2)
        return corr.npairs
    except Exception as exception:
        dask_print(exception)
        raise exception


def join_count_histograms(partial_histograms: list[np.ndarray]) -> np.ndarray:
    """Stack all partial histograms and sum their counts.

    Args:
        partial_histograms (list[np.ndarray]): The list of count histograms
            generated for each pair of partitions.

    Returns:
        The numpy array with the total counts for the partial histograms.
    """
    return np.sum(np.stack(partial_histograms), axis=0)
