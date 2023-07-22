#
import os
from typing import Dict
from .hmm import DataHMM


def collect_fhash(directory: str, /, *, title_observation: str, title_hidden: str) -> Dict[str, Dict[str, str]]:
    R"""
    Collect file hash for a dataset.

    Args
    ----
    - directory
        Dataset directory.
    - title_observation
        Observation directory title.
    - title_hidden
        Hidden state directory title.

    Returns
    -------
    """
    #
    directory_observation = os.path.join(directory, title_observation)
    directory_hidden = os.path.join(directory, title_hidden)
    filenames = DataHMM.collect_indices(directory, title_observation=title_observation, title_hidden=title_hidden)

    #
    fhashes: Dict[str, Dict[str, str]]

    #
    fhashes = {title_observation: {}, title_hidden: {}}
    for name in filenames:
        #
        path_observation = os.path.join(directory_observation, name)
        path_hidden = os.path.join(directory_hidden, name)
        fhashes[title_observation][name] = DataHMM.get_fhash(path_observation)
        fhashes[title_hidden][name] = DataHMM.get_fhash(path_hidden)
    return fhashes
