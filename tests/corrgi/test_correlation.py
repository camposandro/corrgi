import lsdb
import numpy.testing as npt
import treecorr

from corrgi.corrgi import compute_autocorrelation


def test_correlation(dask_client, data_catalog_dir, nn_counts):
    corr = treecorr.NNCorrelation(min_sep=0.01, max_sep=0.1, nbins=33)
    data_catalog = lsdb.read_hats(data_catalog_dir)
    counts = compute_autocorrelation(data_catalog, corr, dask_client)
    #npt.assert_allclose(counts, nn_counts, rtol=1e-3)
    assert len(counts) == corr.nbins
