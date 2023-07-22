#
import pytest
import os
import json
from typing import Dict, Union, Type
from veritas.frameworks import (
    Framework,
    FrameworkCompareHMMCategoricalBasic,
    FrameworkCompareHMMCategoricalStream,
    FrameworkCompareHMMGaussianBasic,
    FrameworkCompareHMMGaussianStream,
    FrameworkFitHMMCategorical,
    FrameworkFitHMMGaussian,
    FrameworkFitHMMStream,
    FrameworkTransformHMMStream,
)


#
with open(os.path.join(os.path.dirname(__file__), "constants.json"), "r") as file:
    #
    CONSTANTS = json.load(file)
LOG_ROOT = CONSTANTS["log_root"]


def framework_compare(kind: str, /) -> Framework:
    R"""
    Framework for comparison.

    Args
    ----
    - kind
        Task kind.

    Returns
    -------
    - framework
        Framework.
    """
    #
    frameworks: Dict[
        str,
        Union[
            Type[FrameworkCompareHMMCategoricalBasic],
            Type[FrameworkCompareHMMCategoricalStream],
            Type[FrameworkCompareHMMGaussianBasic],
            Type[FrameworkCompareHMMGaussianStream],
        ],
    ]

    #
    frameworks = {
        "compare.categorical.basic": FrameworkCompareHMMCategoricalBasic,
        "compare.categorical.stream": FrameworkCompareHMMCategoricalStream,
        "compare.gaussian.basic": FrameworkCompareHMMGaussianBasic,
        "compare.gaussian.stream": FrameworkCompareHMMGaussianStream,
    }
    return frameworks[kind](disk=LOG_ROOT, clean=True)


def framework_fit(kind: str, /) -> Framework:
    R"""
    Framework for fitting.

    Args
    ----
    - kind
        Task kind.

    Returns
    -------
    - framework
        Framework.
    """
    #
    frameworks: Dict[
        str,
        Union[Type[FrameworkFitHMMCategorical], Type[FrameworkFitHMMGaussian], Type[FrameworkFitHMMStream]],
    ]

    #
    frameworks = {
        "fit.categorical": FrameworkFitHMMCategorical,
        "fit.gaussian": FrameworkFitHMMGaussian,
        "fit.stream": FrameworkFitHMMStream,
    }
    return frameworks[kind](disk=LOG_ROOT, clean=True, strict=False)


def framework_transform(kind: str, /) -> Framework:
    R"""
    Framework for transformation.

    Args
    ----
    - kind
        Task kind.

    Returns
    -------
    - framework
        Framework.
    """
    #
    frameworks: Dict[str, Type[FrameworkTransformHMMStream]]

    #
    frameworks = {"transform": FrameworkTransformHMMStream}
    return frameworks[kind](disk=LOG_ROOT, clean=True)


#
PARAMETERS = [
    "compare.categorical.basic",
    "compare.categorical.stream",
    "compare.gaussian.basic",
    "compare.gaussian.stream",
    "fit.categorical",
    "fit.gaussian",
    "fit.stream",
    "transform",
]
FRAMEWORKS = {
    "compare.categorical.basic": framework_compare,
    "compare.categorical.stream": framework_compare,
    "compare.gaussian.basic": framework_compare,
    "compare.gaussian.stream": framework_compare,
    "fit.categorical": framework_fit,
    "fit.gaussian": framework_fit,
    "fit.stream": framework_fit,
    "transform": framework_transform,
}


@pytest.mark.parametrize("kind", PARAMETERS)
def test_annotate(*, kind: str) -> None:
    R"""
    Test framework annotations.

    Args
    ----
    - kind
        Task kind.

    Returns
    -------
    """
    #
    FRAMEWORKS[kind](kind).__annotate__()


def main() -> None:
    R"""
    Main execution.

    Args
    ----

    Returns
    -------
    """
    #
    for kind in PARAMETERS:
        #
        test_annotate(kind=kind)


#
if __name__ == "__main__":
    #
    main()
