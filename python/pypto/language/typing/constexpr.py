# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile-time symbolic values for ``@pl.jit`` templates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstExpr:
    """An unbound compile-time value consumed by ``JITFunction.specialize``."""

    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("constexpr name must be a valid Python identifier")

    def __int__(self) -> int:
        raise TypeError(
            f"constexpr '{self.name}' is unbound; bind it at the JIT call site or call "
            f"kernel.specialize({self.name}=value)"
        )


def constexpr(name: str) -> ConstExpr:
    """Declare a value that must be bound by ``JITFunction.specialize``."""
    return ConstExpr(name)


__all__ = ["ConstExpr", "constexpr"]
