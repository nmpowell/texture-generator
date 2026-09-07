"""Material registry: name -> generator module."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from . import metal, paper, plastic, wood


@runtime_checkable
class Material(Protocol):
    """The contract every material module in :data:`MATERIALS` satisfies.

    ``VARIANTS`` lists the variant names the module accepts, and
    ``generate(shape, rng, variant, **params)`` renders a float32 ``(H, W, 3)``
    array in ``[0, 1]`` for ``shape = (height, width)`` from the given
    ``np.random.Generator``. Modules differ in which keyword parameters they
    accept (see each module's ``generate``), so ``generate`` is typed as a
    general callable rather than with one fixed signature. Both members are
    read-only properties so that a plain module, whose attributes are
    ``list[str]`` and a function, satisfies the protocol structurally, and the
    protocol is runtime-checkable, so ``isinstance(module, Material)`` works.
    """

    @property
    def VARIANTS(self) -> Sequence[str]: ...

    @property
    def generate(self) -> Callable[..., np.ndarray]: ...


MATERIALS: dict[str, Material] = {
    "metal": metal,
    "plastic": plastic,
    "wood": wood,
    "paper": paper,
}

__all__ = ["MATERIALS", "Material", "metal", "paper", "plastic", "wood"]
