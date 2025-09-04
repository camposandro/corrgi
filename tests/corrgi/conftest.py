from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dask.distributed import Client


@pytest.fixture(scope="session", name="dask_client")
def dask_client():
    """Create a single client for use by all unit test cases."""
    client = Client(n_workers=3, threads_per_worker=1)
    yield client
    client.close()


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def hats_catalogs_dir(test_data_dir):
    return test_data_dir / "hats"


@pytest.fixture
def expected_results_dir(test_data_dir):
    return test_data_dir / "expected_results"


@pytest.fixture
def data_catalog_dir(hats_catalogs_dir):
    return hats_catalogs_dir / "DATA"


@pytest.fixture
def rand_catalog_dir(hats_catalogs_dir):
    return hats_catalogs_dir / "RAND"


@pytest.fixture
def single_data_partition(data_catalog_dir):
    return pd.read_parquet(
        data_catalog_dir / "dataset" / "Norder=0" / "Dir=0" / "Npix=1.parquet"
    )


@pytest.fixture
def nn_counts(expected_results_dir):
    return np.load(expected_results_dir / "nn_counts.npy")
