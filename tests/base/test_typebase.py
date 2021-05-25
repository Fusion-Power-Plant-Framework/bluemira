# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

import pytest
import numpy as np
from typing import List, Type, Union

from BLUEPRINT.base.typebase import TypeFrameworkError, TypeBase, Typed
from tests.base.class_habitat import (
    LoopTestClass,
    SystemSystem,
    BreedingBlanket,
    Reactor,
    Exotic,
    TypingListClass,
    TypingUnionListClass,
    TripleUnion,
    function_1,
    function_3,
    function_4,
    function_5,
    function_6,
    function_sizearrays,
    function_arrays,
    function_returncustom,
    function_return_fail1,
    function_return_fail2,
    function_return_fail3,
    function_multi_type,
)
from BLUEPRINT.geometry.loop import Loop


class TestClass:
    @classmethod
    def setup_class(cls) -> None:
        cls.loop = LoopTestClass(1, 2, 3)

    def test_initfail(self):
        fails = [
            [1, 2, 3.0],
            [1.0, 2, 3],
            [1, 2.0, 3],
            [1, 2, "t"],
            [10000000, 200000000000, 2000000.0],
        ]
        for fail in fails:
            with pytest.raises(TypeFrameworkError):
                loop = LoopTestClass(*fail)

    def test_decorator(self):
        self.loop.increment(1)
        with pytest.raises(TypeFrameworkError):
            self.loop.increment("failure")

    def test_partial_annotations(self):
        self.loop.halfannotations(1.4353456, 4)
        with pytest.raises(TypeFrameworkError):
            self.loop.halfannotations(1.3523, 1.0)

    def test_noannotations(self):
        self.loop.noannotations(1.3254534, 3.43265342)

    def test_returnannotations(self):
        self.loop.returnannotations(1, 45356.0)

        with pytest.raises(TypeFrameworkError):
            self.loop.returnannotations(1, 1)


class TestFunctions:
    def test_1(self):

        fails = [[1, 1.0], [1.0, 1], ["t", 11], ["t", "s"], [1.0, 1.0]]

        for fail in fails:
            with pytest.raises(TypeFrameworkError):
                function_1(*fail)

        passes = [[1, 1], [2, 1], [-2, -2], [900000000000000, -80000000000000000000]]
        for passe in passes:
            function_1(*passe)

    def test3(self):
        fails = [
            [-1, 1],
            [-1000000000, 0],
            [0.0, 0.0],
            ["f", "r"],
            ["4", 3],
            ["-4", -10],
        ]
        for fail in fails:
            with pytest.raises(TypeFrameworkError):
                function_3(*fail)

        passes = [[10, -1], [1, 10], [4, -40]]
        for passe in passes:
            function_3(*passe)

    def test_return(self):

        with pytest.raises(TypeFrameworkError):
            function_return_fail1(1, 1)

        with pytest.raises(TypeFrameworkError):
            function_return_fail2(1, 1)

        with pytest.raises(TypeFrameworkError):
            function_return_fail3(1, 1)

    def test_return4(self):

        function_4(4)

        with pytest.raises(TypeFrameworkError):
            function_5(4)

        with pytest.raises(TypeFrameworkError):
            function_6(4)

    def test_returncustom(self):
        loop = function_returncustom(1, 2, 3)

    def test_arrays(self):
        a = np.random.rand(3, 3)
        b = np.random.rand(3, 3)

        function_arrays(a, b)

        with pytest.raises(TypeFrameworkError):
            function_arrays(1, 3)
            function_arrays(a, 3)

    def test_sizedarrays(self):
        a = np.random.rand(3, 4)
        b = np.random.rand(4, 3)
        r = function_sizearrays(a, b)

        with pytest.raises(TypeFrameworkError):
            function_sizearrays(b, a)


class TestClasses:
    def test_reactor(self):
        blanket = BreedingBlanket()
        reactor = Reactor(123, blanket)

        with pytest.raises(TypeFrameworkError):
            Reactor(123, 123)

    def test_static(self):
        blanket = BreedingBlanket()
        blanket.explode()

    def test_bad_static(self):
        blanket = BreedingBlanket()
        with pytest.raises(TypeFrameworkError):
            blanket.work1()

    def test_bad_custom_static(self):
        blanket = BreedingBlanket()
        with pytest.raises(TypeFrameworkError):
            blanket.work2()

    # def test_classmethod(self):
    #     BB = BreedingBlanket()
    #     BB.new_bb()
    #
    # def test_bad_classmethod(self):
    #     BB = BreedingBlanket()
    #     with pytest.raises(TypeFrameworkError):
    #         BB.bad_bb()

    def test_exotic(self):
        exotic = Exotic("goodstring", 0.5, None)

        fails = [
            ["", 0.5, None],
            ["good", 1.2, 4],
            ["good", 1.2, 4.0],
            ["g", -0.000000001, 3.0],
            ["sdgsdhs", 1.1, 35346],
            ["sdg", 0.45, "fail"],
            ["sdg", None, 4],
            [None, None, None],
        ]
        for fail in fails:
            with pytest.raises(TypeFrameworkError):
                Exotic(*fail)


class TestTestSystem:
    @classmethod
    def setup_class(cls) -> None:
        loop = Loop(x=[4, 5, 6, 4], y=[7, 8, 9, 7], z=[0, 0, 0, 0])
        cls.sys = SystemSystem("dummy", 4.0, loop)

    def test_declarations(self):
        with pytest.raises(TypeFrameworkError):
            self.sys.name = 4
        with pytest.raises(TypeFrameworkError):
            self.sys.value = "4"
        self.sys.value = 4
        self.sys.value = 4.0
        self.sys.value = -4.67
        with pytest.raises(TypeFrameworkError):
            self.sys.loop = 4
        self.sys.loop = Loop([4, 5, 6, 7], [5, 6, 7, 8])

        with pytest.raises(TypeFrameworkError):
            self.sys.something = 5

    def test_unioninmethod(self):
        self.sys.do_something(246)
        self.sys.do_something(435.453)

        # bool is a subclass of int, so it is valid to pass True (or False) into a typed
        # function that accepts int
        self.sys.do_something(True)

        with pytest.raises(TypeFrameworkError):
            self.sys.do_something("435rgdfg")

    def test_unionreturn(self):
        with pytest.raises(TypeFrameworkError):
            self.sys.do_something_fail()
        self.sys.do_custom(True)
        self.sys.do_custom(False)
        self.sys.do_custom2(True)
        self.sys.do_custom2(False)


class TestTypeList:
    @classmethod
    def setup_class(cls) -> None:
        cls.sys = TypingListClass()

    def test_declarations(self):
        loop = Loop([4, 5, 6], [6, 7, 8])
        fails = [
            [1, 2, 34, 5, 6, 7.0],
            [1, 2, 34, 5, 6, 7, "dsgdfh"],
            "dsfgfd",
            ["yhrthtr", 4, "dfgdfg"],
            np.array(3),
            "3",
            3.0,
            loop,
            [3.0],
            [3, loop],
            [[3]],
            [loop, 3],
        ]
        for fail in fails:
            with pytest.raises(TypeFrameworkError):
                self.sys.inputs = fail
        self.sys.inputs = [1, 2, 3]

        for fail in fails:
            with pytest.raises(TypeFrameworkError):
                self.sys.outputs = fail

        for fail in fails:
            with pytest.raises(TypeFrameworkError):
                self.sys.custom = fail

        self.sys.outputs = ["1", "two", "Three"]
        self.sys.custom = [loop]

    def test_bad_return(self):
        loops = [Loop([4, 5, 6], [6, 7, 8])]
        with pytest.raises(TypeFrameworkError):
            self.sys.add_loops_bad(loops)

    def test_method(self):
        loops = [Loop([4, 5, 6], [6, 7, 8])]
        self.sys.add_loops(loops)
        with pytest.raises(TypeFrameworkError):
            self.sys.add_loops(Loop([4, 5, 6], [6, 7, 8]))


class TestUnionList:
    @classmethod
    def setup_class(cls):
        cls.sys = TypingUnionListClass()

    def test_class_arguments(self):
        loop = Loop([1, 2, 3], [4, 5, 6])
        fails = [[1.0], 1, loop, [1, 1.0], [Loop, 1]]
        for fail in fails:
            with pytest.raises(TypeFrameworkError):
                self.sys.custom = fail

        for fail in fails:
            with pytest.raises(TypeFrameworkError):
                self.sys.custom2 = fail


def test_triple_union():
    thing = TripleUnion()
    successes = [
        1,
        -1,
        -1.0,
        1.0,
        3.933821654134333e18,
        -3.933821654134333e18,
        "sresgsd",
    ]

    for success in successes:
        thing.value = success

    fails = [[1], Loop([4, 5, 6], [5, 6, 7])]
    for fail in fails:
        with pytest.raises(TypeFrameworkError):
            thing.value = fail

    for success in successes:
        thing.do_something(success)
    for fail in fails:
        with pytest.raises(TypeFrameworkError):
            thing.do_something(fail)

    with pytest.raises(TypeFrameworkError):
        thing.do_something_bad(1)


# THIS IS A KNOWN PROBLEM, BUT I HAVE LOST PATIENCE
# class TestClassVariablesWithArgs(unittest.TestCase):

#     def test_select(self):
#         S = Selection('win')
#         #S = Selection('lose')
#
#     def test_array(self):
#         a = ArrayClass(np.zeros((3,3)))


class ParentClass(Typed):
    pass


class ChildClass(ParentClass):
    pass


class OtherClass:
    pass


class TestInheritedTypes:
    class MyClass(TypeBase):  # noqa(D106)
        inst: Type[ParentClass]

        ParentOrFloat = Union[ParentClass, float]

        def __init__(self, test_instance: Type[ParentClass]) -> None:
            self.inst = test_instance

        def my_func(self, test_instance: Type[ParentClass]) -> Type[ParentClass]:
            return test_instance

        def my_list_func(self, test_list: List[ParentClass]) -> List[ParentClass]:
            return test_list

        def my_union_func(self, test_instance: ParentOrFloat) -> ParentOrFloat:
            return test_instance

    @pytest.mark.parametrize("test_class", [ParentClass, ChildClass])
    def test_class_ok(self, test_class):
        inst = test_class()
        test = TestInheritedTypes.MyClass(inst)

    def test_class_fail(self):
        inst = OtherClass()
        with pytest.raises(TypeFrameworkError):
            test = TestInheritedTypes.MyClass(inst)

    @pytest.mark.parametrize("test_class", [ParentClass, ChildClass])
    def test_func_ok(self, test_class):
        inst = TestInheritedTypes.MyClass(ParentClass())
        test = inst.my_func(test_class())

    def test_func_fail(self):
        inst = TestInheritedTypes.MyClass(ParentClass())
        with pytest.raises(TypeFrameworkError):
            test = inst.my_func(OtherClass())

    @pytest.mark.parametrize("test_class", [ParentClass, ChildClass])
    def test_list_func_ok(self, test_class):
        inst = TestInheritedTypes.MyClass(ParentClass())
        test = inst.my_list_func([test_class()])

    def test_list_func_fail(self):
        inst = TestInheritedTypes.MyClass(ParentClass())
        with pytest.raises(TypeFrameworkError):
            test = inst.my_list_func([OtherClass()])

    @pytest.mark.parametrize("test_val", [ParentClass(), ChildClass(), 1.1])
    def test_union_func_ok(self, test_val):
        inst = TestInheritedTypes.MyClass(ParentClass())
        test = inst.my_union_func(test_val)

    @pytest.mark.parametrize("test_val", [OtherClass(), 1])
    def test_union_func_fail(self, test_val):
        inst = TestInheritedTypes.MyClass(ParentClass())
        with pytest.raises(TypeFrameworkError):
            test = inst.my_union_func(test_val)

    @pytest.mark.parametrize("test_class", [ParentClass, ChildClass])
    def test_set_inst_ok(self, test_class):
        inst = TestInheritedTypes.MyClass(ParentClass())
        inst.inst = test_class()

    def test_set_inst_fail(self):
        inst = TestInheritedTypes.MyClass(ParentClass())
        with pytest.raises(TypeFrameworkError):
            inst.inst = OtherClass()


class TestFuncMultiType:
    """
    A class to tests functions with multiple types.
    """

    def test_function_multi_type(self):
        arr = np.array([1, 2, 3])
        function_multi_type(arr, 1.2)

    @pytest.mark.parametrize("arr,val", [([1, 2, 3], 1.2), (np.array([1, 2, 3]), 1)])
    def test_function_multi_type_bad(self, arr, val):
        with pytest.raises(TypeFrameworkError):
            function_multi_type(arr, val)


if __name__ == "__main__":
    pytest.main([__file__])
