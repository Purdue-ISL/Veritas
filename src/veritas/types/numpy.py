#
import numpy as onp
import numpy.typing as onpt
from typing import Union, Any


#
NPBOOLS = onpt.NDArray[onp.bool8]
NPINTS = onpt.NDArray[onp.int64]
NPFLOATS = onpt.NDArray[onp.float64]
NPNUMS = Union[NPINTS, NPFLOATS]
NPSTRS = onpt.NDArray[onp.str_]
NPRECS = onp.recarray[Any, onp.dtype[onp.void]]
NPANYS = Union[NPBOOLS, NPNUMS, NPSTRS, NPRECS]
