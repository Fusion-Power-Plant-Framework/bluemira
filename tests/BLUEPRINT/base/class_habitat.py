# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

# This is just a place to emulate class definitions in the wild
import numpy as np
from typing import Type, Union, List
from BLUEPRINT.base.typebase import (
    TypeBase,
    typechecked,
    PosInteger,
    NpArray,
    SelectFrom,
    NumberOrNone,
    NonEmptyString,
    Between0and1,
)
from BLUEPRINT.base.baseclass import ReactorSystem
from BLUEPRINT.geometry.loop import Loop


class LoopTestClass(TypeBase):
    x: int
    y: int
    z: int

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def increment(self, offset: int):
        return LoopTestClass(self.x + offset, self.y + offset, self.z + offset)

    def noannotations(self, a, b):
        return self.x + a + b

    def halfannotations(self, a, b: int):
        return self.x + a + b

    def returnannotations(self, a, b) -> float:
        return self.x + a + b


class LoopLoop(TypeBase):
    loop: Type[LoopTestClass]
    name: str

    def __init__(self, loop, name):
        self.loop = loop
        self.name = name


class ReactorSystem22(TypeBase):
    CADConstructor = NotImplemented

    def __init_subclass__(cls):
        cls.p = {}
        super().__init_subclass__()

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls)
        self.geom = {}
        self.requirements = {}
        return self


class Exotic(ReactorSystem22):
    name: NonEmptyString
    value: Between0and1
    stat: NumberOrNone

    def __init__(self, name: NonEmptyString, value: Between0and1, stat: NumberOrNone):
        self.name = name
        self.value = value
        self.stat = stat


class Selection(ReactorSystem22):
    choice: SelectFrom("win", "lose")  # noqa

    def __init__(self, choice):
        self.choice = choice


class ArrayClass(ReactorSystem22):

    x: NpArray(3, 3)

    def __init__(self, x):
        self.x = np.zeros((3, 3))


class BreedingBlanket(ReactorSystem22):
    def __init__(self):
        pass

    @staticmethod
    def explode() -> str:
        return "Oh no..."

    @staticmethod
    def work1() -> float:
        return -1

    @staticmethod
    def work2() -> PosInteger:
        return -1

    def normal(self, a: int) -> int:
        return 2 * a

    # Forward ref problme
    # @classmethod
    # def new_bb(cls) -> Type['BreedingBlanket']:
    #     return cls()
    #
    # @classmethod
    # def bad_bb(cls) -> Type['BreedingBlanket']:
    #     return 0


class Reactor(ReactorSystem22):
    xyz: int
    BB: Type[BreedingBlanket]

    def __init__(self, xyz: int, bb: Type[BreedingBlanket]):
        self.xyz = xyz
        self.BB = bb


class DUmmy:
    pass


class SystemSystem(ReactorSystem):
    name: str
    value: Union[int, float]
    default: str = "wtf"
    loop: Type[Loop]
    random = "WTRGDFGH"
    something: Type[DUmmy]
    # COmment

    p = [
        ["P_fus", "Total fusion power", 2000, "MW", None, "PLASMOD"],
        ["P_fus_DT", "D-T fusion power", 1995, "MW", None, "PLASMOD"],
    ]

    def __init__(self, name: str, value, loop: Type[Loop]) -> None:
        self.name = name
        self.loop = loop
        self.value = value
        self.something = DUmmy()

    def do_something(self, do: Union[int, float]):
        if do:
            return 1
        else:
            return 2.0

    def do_something_fail(self) -> Union[int, Type[Loop]]:
        return 5.7687698

    def do_custom(self, do: bool) -> Union[int, None]:
        if do:
            return 1
        else:
            return None

    def do_custom2(self, do: bool) -> Union[Type[Loop], int]:
        if do:
            return Loop(x=[4, 5, 6, 4], y=[7, 8, 9, 9])
        else:
            return 1


class ClassWithDEfaults(ReactorSystem):
    name: str = "DEFAULT"

    def do_something(self, do: bool = True):
        if do:
            print("y")
        else:
            print("n")


class TypingListClass(ReactorSystem):
    somthing: str
    inputs: List[int]
    outputs: List[str]
    custom: List[Type[Loop]]

    def __init__(self):
        pass

    def add_loops(self, loops: List[Type[Loop]]) -> List[int]:
        self.custom = loops
        return [1, 2, 3]

    def add_loops_bad(self, loops: List[Type[Loop]]) -> List[int]:
        self.custom = loops
        return [1, 2, 3.0]


class TypingUnionListClass(ReactorSystem):
    custom: Union[List[Type[Loop]], List[int]]
    custom2: Union[List[Type[Loop]], List[Type[BreedingBlanket]]]

    def do_custom(
        self, input: Union[List[Type[Loop]], List[int]]
    ) -> Union[List[Type[Loop]], List[int]]:
        return input

    def do_custom2(
        self, input: Union[List[Type[Loop]], List[Type[BreedingBlanket]]]
    ) -> Union[List[Type[Loop]], List[Type[BreedingBlanket]]]:
        return input


class TripleUnion(ReactorSystem):
    value: Union[str, int, float]

    def do_something(self, input: Union[str, int, float]) -> Union[str, int, float]:
        return input

    def do_something_bad(self, input: Union[str, int, float]) -> Union[str, int, float]:
        return ["bad"]


@typechecked
def function_1(a: int, b: int) -> int:
    return a + b


@typechecked
def function_2(a: int, b: int) -> (int, int):
    return a + b, a - b


@typechecked
def function_3(a: PosInteger, b: int) -> (int, int):
    return a + b, a - b


@typechecked
def function_4(a: int) -> (int, PosInteger):
    return a, abs(a)


@typechecked
def function_5(a: int) -> (int, PosInteger):
    return a, abs(a) + 0.1


@typechecked
def function_6(a: int) -> (int, PosInteger):
    return a, -a


@typechecked
def function_custom(a: float, b: Type[LoopTestClass]) -> bool:
    a *= b.x
    return True


@typechecked
def function_noreturn(a: float):
    pass


@typechecked
def function_returncustom(a: int, b: int, c: int) -> Type[LoopTestClass]:
    return LoopTestClass(a, b, c)


@typechecked
def function_arrays(a: NpArray, b: NpArray) -> NpArray:
    return a + b


@typechecked
def function_sizearrays(a: NpArray(3, 4), b: NpArray(4, 3)) -> NpArray(3, 3):
    c = np.dot(a, b)
    return c


@typechecked
def function_return_fail1(a: int, b: int) -> (int, float):
    a *= b
    return a + 0.1, a + 0.1


@typechecked
def function_return_fail2(a: int, b: int) -> (int, float):
    a *= b
    return a, a


@typechecked
def function_return_fail3(a: int, b: int) -> (int, float):
    a *= b
    return a + 0.1, a


@typechecked
def function_custom_union(
    input: Union[List[Type[Loop]], List[Type[BreedingBlanket]]]
) -> Union[List[Type[Loop]], List[Type[BreedingBlanket]]]:
    return input


@typechecked
def function_multi_type(arr: NpArray, val: float):
    return arr * val


if __name__ == "__main__":
    pass
