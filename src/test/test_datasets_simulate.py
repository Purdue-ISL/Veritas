#
import pytest
import numpy as onp
import os
import shutil
import json
from typing import Callable
from veritas.datasets import DataHMMCategorical, DataHMMGaussian, DataHMM
from veritas.types import NPINTS


#
with open(os.path.join(os.path.dirname(__file__), "constants.json"), "r") as file:
    #
    CONSTANTS = json.load(file)
DATA_ROOT = CONSTANTS["data_root"]


#
DATA_DIR0 = os.path.join(DATA_ROOT, "__Test__")


def categorical(sizes: NPINTS, rng: onp.random.RandomState, /) -> DataHMM:
    R"""
    Categorical HMM parameters.

    Args
    ----
    - sizes
        Sample sizes.
    - rng
        Random states.

    Returns
    -------
    - dataset
        Dataset.
    """
    #
    num_hiddens = 4
    num_observations = num_hiddens + 2

    #
    initials_ele0 = onp.ones((num_hiddens,), dtype=onp.float64)

    #
    transitions_ele0 = onp.diagflat(onp.ones((num_hiddens - 1,), dtype=onp.float64), 1)
    transitions_ele1 = onp.diagflat(onp.ones((1,), dtype=onp.float64), 1 - num_hiddens)

    #
    emissions_ele0 = onp.diagflat(onp.ones((num_observations,), dtype=onp.float64), 0) * 8
    emissions_ele1 = onp.diagflat(onp.ones((num_observations - 1,), dtype=onp.float64), 1)
    emissions_ele2 = onp.diagflat(onp.ones((num_observations - 1,), dtype=onp.float64), -1)

    #
    initials = initials_ele0
    transitions = transitions_ele0 + transitions_ele1
    emissions = (emissions_ele0 + emissions_ele1 + emissions_ele2)[1:-1]

    #
    initials /= onp.sum(initials)
    transitions /= onp.sum(transitions, axis=1, keepdims=True)
    emissions /= onp.sum(emissions, axis=1, keepdims=True)
    return DataHMMCategorical.from_simulate(initials, transitions, emissions, sizes, random_state=rng)


def gaussian1d(sizes: NPINTS, rng: onp.random.RandomState, /) -> DataHMM:
    R"""
    1D-Gaussian HMM parameters.

    Args
    ----
    - sizes
        Sample sizes.
    - rng
        Random states.

    Returns
    -------
    - dataset
        Dataset.
    """
    #
    num_hiddens = 4

    #
    initials_ele0 = onp.ones((num_hiddens,), dtype=onp.float64)

    #
    transitions_ele0 = onp.diagflat(onp.ones((num_hiddens - 1,), dtype=onp.float64), 1)
    transitions_ele1 = onp.diagflat(onp.ones((1,), dtype=onp.float64), 1 - num_hiddens)

    #
    means0 = onp.arange(num_hiddens).astype(onp.float64)
    covars0 = (1 + onp.arange(num_hiddens)).astype(onp.float64) * 0.05

    #
    initials = initials_ele0
    transitions = transitions_ele0 + transitions_ele1
    means = onp.reshape(means0, (num_hiddens, 1))
    covars = onp.reshape(covars0, (num_hiddens, 1))

    #
    initials /= onp.sum(initials)
    transitions /= onp.sum(transitions, axis=1, keepdims=True)
    return DataHMMGaussian.from_simulate(initials, transitions, means, covars, sizes, random_state=rng)


def gaussian2d(sizes: NPINTS, rng: onp.random.RandomState, /) -> DataHMM:
    R"""
    2D-Gaussian HMM parameters.

    Args
    ----
    - sizes
        Sample sizes.
    - rng
        Random states.

    Returns
    -------
    - dataset
        Dataset.
    """
    #
    num_hiddens = 4

    #
    initials_ele0 = onp.ones((num_hiddens,), dtype=onp.float64)

    #
    transitions_ele0 = onp.diagflat(onp.ones((num_hiddens - 1,), dtype=onp.float64), 1)
    transitions_ele1 = onp.diagflat(onp.ones((1,), dtype=onp.float64), 1 - num_hiddens)

    #
    means0 = onp.arange(num_hiddens).astype(onp.float64)
    covars0 = (1 + onp.arange(num_hiddens)).astype(onp.float64) * 0.05
    means1 = onp.flip(means0)
    covars1 = onp.flip(covars0)

    #
    initials = initials_ele0
    transitions = transitions_ele0 + transitions_ele1
    means = onp.stack((means0, means1), axis=1)
    covars = onp.stack((covars0, covars1), axis=1)

    #
    initials /= onp.sum(initials)
    transitions /= onp.sum(transitions, axis=1, keepdims=True)
    return DataHMMGaussian.from_simulate(initials, transitions, means, covars, sizes, random_state=rng)


#
PARAMETERS = [(categorical, 100, 40), (gaussian1d, 100, 40), (gaussian2d, 100, 40)]


@pytest.mark.parametrize(("prior", "nsteps", "num"), PARAMETERS)
def test_simulate(*, prior: Callable[[NPINTS, onp.random.RandomState], DataHMM], nsteps: int, num: int) -> None:
    R"""
    Test simulation.

    Args
    ----
    - prior
        Prior knowledge.
    - nsteps
        Expected number of steps per simulating trace.
    - num
        Number of simulating traces.

    Returns
    -------
    """
    #
    if os.path.isdir(DATA_DIR0):
        #
        shutil.rmtree(DATA_DIR0)
    os.makedirs(DATA_DIR0)

    #
    rng = onp.random.RandomState(42)
    sizes = rng.randint(int(nsteps * 0.95), int(nsteps * 1.05), (num,))
    prior(sizes, rng).save_csv(DATA_DIR0, title_observation="observation", title_hidden="hidden")


def test_clear() -> None:
    R"""
    Clear cache as test.

    Args
    ----

    Returns
    -------
    """
    #
    if os.path.isdir(DATA_DIR0):
        #
        shutil.rmtree(DATA_DIR0)


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    for (prior, nsteps, num) in PARAMETERS:
        #
        test_simulate(prior=prior, nsteps=nsteps, num=num)
    test_clear()


#
if __name__ == "__main__":
    #
    main()
