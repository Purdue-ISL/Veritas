#
import os
import json
import numpy as onp
from veritas.frameworks import FrameworkLoadHMMStream
from veritas.datasets.hmm.stream import get_capacities_discretized, get_capacities_continuous


#
with open(os.path.join(os.path.dirname(__file__), "constants.json"), "r") as file:
    #
    CONSTANTS = json.load(file)
DATA_ROOT = CONSTANTS["data_root"]
LOG_ROOT = CONSTANTS["log_root"]


#
DATA_DIR0 = os.path.join(DATA_ROOT, "Stream2")


def test_discrete() -> None:
    R"""
    Test capacicty discretization.

    Args
    ----

    Returns
    -------
    """
    #
    args = []
    args.extend(["--dataset", DATA_DIR0])
    args.extend(["--capacity-max", "10.0"])
    args.extend(["--filter-capmax"])

    #
    framework = FrameworkLoadHMMStream(disk=LOG_ROOT, clean=True)
    framework.phase1(args)
    framework.erase()

    # Capacity unit is 0.5 Mbps by default.
    capunit = 0.5

    # Test capacity discretization.
    num_hiddens = 1 + int(round(10.0 / capunit))
    capacities_discretize = get_capacities_discretized(framework._dataset_full, capunit)
    capacities_continuous = get_capacities_continuous(framework._dataset_full)

    #
    capmax_discretized = max(onp.max(it_discretize).item() for it_discretize in capacities_discretize)
    capmax_expect = (num_hiddens - 1) * capunit
    assert capmax_discretized == capmax_expect

    #
    for dx in [-1, 1]:
        #
        for (it_discretized, it_continuous) in zip(capacities_discretize, capacities_continuous):
            #
            assert onp.all(
                onp.abs(it_discretized - it_continuous) <= onp.abs(it_discretized + dx * capunit - it_continuous),
            )


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    test_discrete()


#
if __name__ == "__main__":
    #
    main()
