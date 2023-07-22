#
from veritas.datasets import DataHMM


def test_null() -> None:
    R"""
    Test null file hashing.

    Args
    ----

    Returns
    -------
    """
    #
    assert DataHMM.get_fhash("") == ""


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    test_null()


#
if __name__ == "__main__":
    #
    main()
