import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import cleaning, features, io_utils, merge, target  # noqa: E402


@pytest.fixture(scope="session")
def raw():
    return io_utils.load_raw_tables()


@pytest.fixture(scope="session")
def clean(raw):
    return cleaning.clean_tables(raw)[0]


@pytest.fixture(scope="session")
def merged(clean):
    return merge.build_order_table(clean)[0]


@pytest.fixture(scope="session")
def full(merged):
    f = target.add_target(features.build_features(merged))
    funnel, mask = target.eligibility_funnel(f)
    f["eligible"] = mask
    f.attrs["funnel"] = funnel
    return f


@pytest.fixture(scope="session")
def model_table(full):
    return full[full.eligible].reset_index(drop=True)
