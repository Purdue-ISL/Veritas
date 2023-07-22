#
from __future__ import annotations


#
import abc
import numpy as onp
import os
import time
import shutil
import json
import argparse
from datetime import datetime
from typing import Sequence
from ..types import NPNUMS


class Framework(abc.ABC):
    R"""
    Framework to execute algorithm(s) from terminal.
    """

    def __annotate__(self: Framework) -> None:
        R"""
        Annotations.

        Args
        ----

        Returns
        -------
        """
        #
        self._disk: str
        self._clean: bool

        #
        self._parser: argparse.ArgumentParser
        self._title_suffix: str
        self._eq_eps: float

    @abc.abstractmethod
    def arguments(self: Framework, /) -> None:
        R"""
        Define argument(s).

        Args
        ----

        Returns
        -------
        """
        #
        pass

    def framein(self: Framework, cmds: Sequence[str], /) -> None:
        R"""
        Get arguments from framework input interface.

        Args
        ----
        - cmds
            Commands.

        Returns
        -------
        """
        #
        self._args = self._parser.parse_args() if len(cmds) == 0 else self._parser.parse_args(cmds)

    @abc.abstractmethod
    def parse(self: Framework, /) -> None:
        R"""
        Parse argument(s) from given command(s).

        Args
        ----

        Returns
        -------
        """
        #
        pass

    def logs(self: Framework, /) -> None:
        R"""
        Prepare logging space(s).

        Args
        ----

        Returns
        -------
        """
        #
        self._title = self.identifier()
        self._disk_log = os.path.join(self._disk, self._title)
        os.makedirs(self._disk_log)

        #
        with open(os.path.join(self._disk_log, "FRAMEWORK.txt"), "w") as file:
            #
            file.write(self.__class__.__name__)
        with open(os.path.join(self._disk_log, "arguments.json"), "w") as file:
            #
            json.dump(self._args.__dict__, file, indent=4)

    def identifier(self: Framework, /) -> str:
        R"""
        Create a disk-unique identifier.

        Args
        ----

        Returns
        -------
        - name
            A disk-unique identifier.
        """
        #
        while True:
            #
            time.sleep(1.0)
            now = datetime.now()
            name = "{:04d}{:02d}{:02d}{:02d}{:02d}{:02d}".format(
                int(now.year),
                int(now.month),
                int(now.day),
                int(now.hour),
                int(now.minute),
                int(now.second),
            )
            if len(self._title_suffix) > 0:
                #
                name = "{:s}:{:s}".format(name, self._title_suffix)
            directory = os.path.join(self._disk, name)
            if not os.path.isdir(directory):
                #
                break
        return name

    @abc.abstractmethod
    def datasets(self: Framework, /) -> None:
        R"""
        Prepare dataset(s).

        Args
        ----

        Returns
        -------
        """
        #
        pass

    @abc.abstractmethod
    def models(self: Framework, /) -> None:
        R"""
        Prepare model(s).

        Args
        ----

        Returns
        -------
        """
        #
        pass

    @abc.abstractmethod
    def loaders(self: Framework, /) -> None:
        R"""
        Prepare loader(s).

        Args
        ----

        Returns
        -------
        """
        #
        pass

    @abc.abstractmethod
    def algorithms(self: Framework, /) -> None:
        R"""
        Prepare algorithm(s).

        Args
        ----

        Returns
        -------
        """
        #
        pass

    @abc.abstractmethod
    def execute(self: Framework, /) -> None:
        R"""
        Execute.

        Args
        ----

        Returns
        -------
        """
        #
        pass

    def erase(self: Framework, /) -> None:
        R"""
        Erase logging space(s).

        Args
        ----

        Returns
        -------
        """
        #
        shutil.rmtree(self._disk_log)

    def __call__(self: Framework, cmds: Sequence[str], /) -> None:
        R"""
        Call the class.

        Args
        ----
        - cmds
            Command(s) for execution.

        Returns
        -------
        """
        #
        self.phase4(cmds)

    def phase0(self: Framework, cmds: Sequence[str], /) -> None:
        R"""
        Phase 0 of calling the class.

        Args
        ----
        - cmds
            Command(s) for execution.

        Returns
        -------
        """
        #
        self.arguments()
        self.framein(cmds)
        self.parse()
        self.logs()

    def phase1(self: Framework, cmds: Sequence[str], /) -> None:
        R"""
        Phase 1 of calling the class.

        Args
        ----
        - cmds
            Command(s) for execution.

        Returns
        -------
        """
        #
        self.phase0(cmds)
        self.datasets()

    def phase2(self: Framework, cmds: Sequence[str], /) -> None:
        R"""
        Phase 2 of calling the class.

        Args
        ----
        - cmds
            Command(s) for execution.

        Returns
        -------
        """
        #
        self.phase1(cmds)
        self.models()

    def phase3(self: Framework, cmds: Sequence[str], /) -> None:
        R"""
        Phase 3 of calling the class.

        Args
        ----
        - cmds
            Command(s) for execution.

        Returns
        -------
        """
        #
        self.phase2(cmds)
        self.loaders()
        self.algorithms()

    def phase4(self: Framework, cmds: Sequence[str], /) -> None:
        R"""
        Phase 4 of calling the class.

        Args
        ----
        - cmds
            Command(s) for execution.

        Returns
        -------
        """
        #
        self.phase3(cmds)
        self.execute()
        if self._clean:
            #
            self.erase()

    def equal(self: Framework, array1: NPNUMS, array2: NPNUMS, /) -> bool:
        R"""
        Check if two arraies are equal.

        Args
        ----
        - array1
            Testing array.
        - array2
            Baseline array.

        Returns
        -------
        - flag
            If True, two arraies are equal.
        """
        #
        if issubclass(array1.dtype.type, onp.floating):
            #
            array1 = onp.minimum(onp.maximum(array1, onp.finfo(array1.dtype).min), onp.finfo(array1.dtype).max)
        if issubclass(array2.dtype.type, onp.floating):
            #
            array2 = onp.minimum(onp.maximum(array2, onp.finfo(array2.dtype).min), onp.finfo(array2.dtype).max)
        return float(onp.max(onp.abs(array1 - array2)).item()) < self._eq_eps
