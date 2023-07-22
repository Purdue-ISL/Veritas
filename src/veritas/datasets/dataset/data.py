#
from __future__ import annotations


#
import more_itertools as xitertools
import os
import hashlib
from typing import Dict, Type, TypeVar, List
from .meta import MetaFinite


#
AnySample = TypeVar("AnySample")


class DataFinite(MetaFinite[AnySample]):
    R"""
    Concrete data of finite samples.
    """
    #
    TYPES: Dict[str, List[str]]

    #
    IOCHUNK = 1024**2

    def repr_columns(self: DataFinite[AnySample], /, *, indent: int) -> str:
        R"""
        Get column definition representation.

        Args
        ----
        - indent
            Number of indent spacing per representation line.

        Returns
        -------
        - representation
            Representation.
        """
        #
        maxlen1 = len(str(len(self.columns)))
        maxlen2 = max(
            xitertools.collapse(
                [[len(name) for (name, names) in record.items() if len(names) > 0] for record in self.columns],
            ),
        )
        maxlen3 = max(len(name) for name in xitertools.collapse([list(record.values()) for record in self.columns]))

        #
        lines = []
        for (i1, record) in enumerate(self.columns):
            #
            for (i2, (name2, name3s)) in enumerate(
                (name2, name3s) for (name2, name3s) in record.items() if len(name3s) > 0
            ):
                #
                if i2 == 0:
                    #
                    line = "+-{:s}-+-{:s}-+-{:s}-+".format("-" * maxlen1, "-" * maxlen2, "-" * maxlen3)
                else:
                    #
                    line = "| {:s} +-{:s}-+-{:s}-+".format(" " * maxlen1, "-" * maxlen2, "-" * maxlen3)
                lines.append(" " * indent + line)
                for (i3, name3) in enumerate(name3s):
                    #
                    name1 = str(i1) if i3 == 0 and i2 == 0 else ""
                    name2 = name2 if i3 == 0 else ""
                    line = "| {:{:d}s} | {:{:d}s} | {:{:d}s} |".format(name1, maxlen1, name2, maxlen2, name3, maxlen3)
                    lines.append(" " * indent + line)
        line = "+-{:s}-+-{:s}-+-{:s}-+".format("-" * maxlen1, "-" * maxlen2, "-" * maxlen3)
        lines.append(" " * indent + line)
        return "\n".join(lines)

    @classmethod
    def validate_path(cls: Type[DataFinite[AnySample]], path: str, fhash: str, /) -> None:
        R"""
        Validate loading path.

        Args
        ----
        - path
            Path.
        - fhash
            File hash.

        Returns
        -------
        """
        #
        msg = cls.check_fhash(cls.get_fhash(path), fhash)
        if msg:
            #
            raise RuntimeError(msg)

    @classmethod
    def get_fhash(cls: Type[DataFinite[AnySample]], path: str, /) -> str:
        R"""
        Get file hash.

        Args
        ----
        - path
            Path.

        Returns
        -------
        - fhash
            File hash.
        """
        #
        if not os.path.isfile(path):
            #
            return ""
        else:
            #
            with open(path, "rb") as file:
                #
                hunit = hashlib.sha256()
                while chunk := file.read(cls.IOCHUNK):
                    #
                    hunit.update(chunk)
            return hunit.hexdigest()

    @staticmethod
    def check_fhash(test: str, true: str, /) -> str:
        R"""
        Check file hash.

        Args
        ----
        - test
            Testing file hash.
        - true
            True file hash.

        Returns
        -------
        - msg
            Checking message.
        """
        #
        if true == "*":
            #
            return (
                'True file hash is \x1b[92mwildcard\x1b[0m, while testing file hash is "\x1b[93m{:s}\x1b[0m"'.format(
                    test,
                )
            )
        elif test != true:
            #
            return 'True file hash is "\x1b[92m{:s}\x1b[0m", while testing file hash is "\x1b[93m{:s}\x1b[0m"'.format(
                true,
                test,
            )
        else:
            #
            return ""

    def __len__(self: DataFinite[AnySample], /) -> int:
        R"""
        Get length,

        Args
        ----

        Returns
        -------
        - length
            Length.
        """
        #
        return max(len(self.indices), len(self.sections))
