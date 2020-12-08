import pytest

from hbayesdm.models import generalise_gs


def test_generalise_gs():
    _ = generalise_gs(
        data="example", niter=10, nwarmup=5, nchain=1, ncore=1)


if __name__ == '__main__':
    pytest.main()
