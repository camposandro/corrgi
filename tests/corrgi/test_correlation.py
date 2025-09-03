import hats
import numpy.testing as npt
import treecorr

from corrgi.corrgi import compute_autocorrelation, count_auto_pairs


def test_correlation(dask_client, data_catalog, nn_counts):
    corr = treecorr.NNCorrelation(min_sep=0.01, max_sep=0.1, nbins=33)
    counts = compute_autocorrelation(data_catalog, corr).compute()
    npt.assert_allclose(counts, nn_counts, rtol=1e-3)


def test_count_auto_pairs(single_data_partition, data_catalog_dir):
    corr = treecorr.NNCorrelation(min_sep=0.01, max_sep=0.1, nbins=33)
    data_catalog = hats.read_hats(data_catalog_dir)
    counts = count_auto_pairs(single_data_partition, data_catalog.catalog_info, corr).compute()
    assert len(counts) == corr.nbins
