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

"""
The home of the typechecking framework

All of this stuff is heavily inspired from some of David Beazley's work and
is dangerous and experimental as fuck. Fucking love it.
"""
from typing import Type, Union, List
import numpy as np
from functools import wraps
from inspect import signature

# Do not use anything else in here! .. OK, so you _can_ use the Contracts in
# functions..
__all__ = [
    "TypeBase",
    "typechecked",
    "ENGAGE_TYPECHECKING",
    "TypeFrameworkError",
    "Contract",
]

# Whether or not to enable type-checking
ENGAGE_TYPECHECKING = True


# Dictionary for automatic registration of contracts
CONTRACTS = {}

# TODO: Defaults specified in class declaration or function annotations
# currently not addressed
# TODO: Contracts with arguments cannot be specified in class declarations


class TypeFrameworkError(Exception):
    """
    The Error class for the typing framework
    """

    pass


class Contract:
    """
    The base class for enforcing types and rules
    """

    def __init_subclass__(cls):
        """
        Captures information of contracts upon instantiation
        """
        CONTRACTS[cls.__name__] = cls

    def __set__(self, instance, value, *args):
        """
        Set the value of an attribute onto an instance.

        Parameters
        ----------
        instance: object
            The instance of the object upon which to set a value
        value: Any
            The value to be checked and assigned by the Contract
        """
        self._verify(value, *args)
        instance.__dict__[self.name] = value

    def __set_name__(self, cls, name):
        """
        Set the name of the Contract
        """
        self.name = name

    @classmethod
    def _verify(cls, value, *args):
        """
        Checks parameters against their specified rules. It is also present in
        the base-class, hence why it have a secretive name.
        """
        pass


class Typed(Contract):
    """
    The base class for simple type-checking
    """

    ty = object

    @classmethod
    def _verify(cls, value, *args):
        if not isinstance(value, cls.ty):
            raise TypeFrameworkError(f"Expected {cls.ty}")
        super()._verify(value)


class TypedList(Contract):
    """
    Class for checking types inside a list
    """

    ty = None

    @classmethod
    def _verify(cls, value, *args):
        if not isinstance(value, list):
            raise TypeFrameworkError
        for val in value:
            if not isinstance(val, cls.ty):
                raise TypeFrameworkError


class CustomTypedList(Contract):
    """
    Class for checking custom types inside a list
    """

    ty = None

    @classmethod
    def _verify(cls, value, *args):
        if not isinstance(value, list):
            raise TypeFrameworkError
        for val in value:
            if not isinstance(val, cls.ty):
                raise TypeFrameworkError


class Integer(Typed):
    ty = int


class Float(Typed):
    ty = float


class String(Typed):
    ty = str


class Dict(Typed):
    ty = dict


class ListList(Typed):
    ty = list


class Tuple(Typed):
    ty = tuple


class Boolean(Typed):
    ty = bool


class NoneTyper(Typed):
    ty = type(None)


class NpArray(Contract):
    __name__ = "NpArray"
    ty = np.ndarray

    _args = (None, None)

    def __init__(self, n=None, m=None):
        self.n = n
        self.m = m
        self._args = (n, m)

    @classmethod
    def _verify(cls, value, args):
        if not isinstance(value, cls.ty):
            raise TypeFrameworkError
        if args == (None, None):
            pass

        else:
            if not value.shape == args:
                raise TypeFrameworkError


class SelectFrom(Contract):
    __name__ = "SelectFrom"
    _args = []

    def __init__(self, *args):
        self._args = args

    @classmethod
    def _verify(cls, value, args):
        if value not in args:
            raise TypeFrameworkError


# Dictionary mapping for built-in types and their associated Contracts
BUILTIN_MAP = {
    int: Integer,
    float: Float,
    str: String,
    list: ListList,
    dict: Dict,
    bool: Boolean,
    tuple: Tuple,
    None: NoneTyper,
    type(None): NoneTyper,
}


class Number(Typed):
    ty = float, int


class NumberOrNone(Typed):
    ty = float, int, type(None)


class Positive(Contract):
    @classmethod
    def _verify(cls, value, *args):
        if value < 0:
            raise TypeFrameworkError("Expected >= 0")
        super()._verify(value)


class NonEmpty(Contract):
    @classmethod
    def _verify(cls, value, *args):
        if not len(value) > 0:
            raise TypeFrameworkError
        super()._verify(value)


class NonEmptyString(String, NonEmpty):
    pass


class PosInteger(Integer, Positive):
    pass


class PosFloat(Float, Positive):
    pass


class Between0and1(Number):
    @classmethod
    def _verify(cls, value, *args):
        super()._verify(value)
        if not 0 <= value <= 1:
            raise TypeFrameworkError


class CustomTyped(Contract):
    ty = None

    @classmethod
    def _verify(cls, value, *args):

        if not issubclass(value.__class__, cls.ty):
            raise TypeFrameworkError(f"Expected {cls.ty.__name__}")


class CustomUnionTyped(Contract):
    ty = None

    @classmethod
    def _verify(cls, value, *args):
        fail = True
        for ty in cls.ty:
            if isinstance(ty, str):
                if ty == value.__class__.__name__:
                    fail = False

            elif hasattr(ty, "__origin__"):
                if ty.__origin__ is List or ty.__origin__ is list:
                    if not isinstance(value, list):
                        raise TypeFrameworkError

                    list_type = ty.__args__[0]
                    for val in value:

                        if not list_type == val.__class__.__name__:
                            raise TypeFrameworkError

            elif isinstance(value, ty):
                fail = False

        if fail:
            raise TypeFrameworkError


class UnionContract(Contract):
    @classmethod
    def _verify(cls, value, *args):

        for contract in cls.__mro__[1:-2]:
            try:
                contract._verify(value)
                break

            except TypeFrameworkError:
                continue
        else:
            raise TypeFrameworkError


def make_typing_contract(typ):
    """
    Makes a Contract class for a typing Type

    Parameters
    ----------
    typ: typing.Type

    Makes a class, auto-populating the CONTRACTS list
    """
    type(typ.__args__[0].__name__, (CustomTyped,), {"ty": typ.__args__[0]})


def make_union_contract(typ, name):
    """
    Makes a Contract class for a typing Union

    Parameters
    ----------
    typ: typing.Union
        The Union for which to create the contract
    name: str
        The name of the contract to create

    Makes a class, auto-populating the CONTRACTS list
    """
    args = []
    for arg in typ.__args__:
        if hasattr(arg, "__origin__") and (
            arg.__origin__ is Type or arg.__origin__ is type
        ):
            arg = arg.__args__[0]
        if hasattr(arg, "__name__") and arg.__name__ == "NoneType":
            args.append(type(None))
        elif arg in BUILTIN_MAP:
            args.append(arg)
        elif hasattr(arg, "__origin__"):
            if arg.__origin__ is List or arg.__origin__ is list:
                pass

            elif arg.__origin__ is Type or arg.__origin__ is type:
                args.append(arg.__args__[0])

            else:
                raise TypeFrameworkError("Da ist etwas shief gelaufen...")

        else:
            args.append(arg)

    type(name, (CustomUnionTyped,), {"ty": args})


def make_list_contract(typ, name):
    """
    Makes a Contract class for a typing List

    Parameters
    ----------
    typ: typing.List
        The List for which to create a contract
    name: str
        The name of the contract to create

    Makes a class, auto-populating the CONTRACTS list
    """
    arg = typ.__args__[0]
    base_class = TypedList
    if hasattr(arg, "__name__") and arg.__name__ == "NoneType":
        arg = type(None)
    elif arg in BUILTIN_MAP:
        arg = arg

    elif hasattr(arg, "__origin__"):
        if arg.__origin__ is Type or arg.__origin__ is type:
            base_class = CustomTypedList
            arg = arg.__args__[0]
    else:
        base_class = CustomTypedList
        arg = arg

    type(name, (base_class,), {"ty": arg})


def _get_type_names(typ_args):
    """
    Stitch the names of the underlying types together

    Parameters
    ----------
    args: List[Type[Any]]
        List of type arguments for a compound type e.g. Union or List

    Returns
    -------
    type_names: str
        The string representing the underlying types

    Examples
    --------
    >>> from typing import List, Type, Union
    >>> from BLUEPRINT.geometry.loop import Loop
    >>> typ = Union[Type[Loop], List[float]]
    >>> print(f"Union{_get_type_names(typ.__args__)}")
    UnionLoopListFloat
    """
    result = ""
    for arg in typ_args:
        if hasattr(arg, "__args__"):
            if hasattr(arg, "__origin__") and arg.__origin__ is not type:
                result += arg.__origin__.__name__.capitalize()
            result += _get_type_names(arg.__args__)
        else:
            while not hasattr(arg, "__name__"):
                result += _get_type_names(arg.__args__)
            else:
                result += arg.__name__.capitalize()
    return result


def get_contract(typ):
    """
    Gets the contract for a specified type.

    Parameters
    ----------
    typ: builtin or class
        The object for which to return the contract

    Returns
    -------
    contract: Contract
        The contract of the specified type

    Raises
    ------
    TypeFrameworkError:
        If the input typ is an acceptable contract
    """
    contract = None

    if typ in BUILTIN_MAP:
        contract = BUILTIN_MAP[typ]()

    if hasattr(typ, "__name__"):
        if typ.__name__ in CONTRACTS:
            contract = CONTRACTS[typ.__name__]()

        elif hasattr(typ, "_gorg") and hasattr(typ, "__args__"):
            # Handle typing.List
            if hasattr(typ, "__origin__"):
                if typ.__origin__ is List or typ.__origin__ is list:
                    # Handle typing.List object in Python 3.6
                    name = "List" + typ.__args__[0].__name__.capitalize()
                    if name not in CONTRACTS:
                        make_list_contract(typ, name)
                    contract = CONTRACTS[name]()
                else:
                    # Handle typing.Type object in Python 3.6
                    make_typing_contract(typ)
                    contract = CONTRACTS[typ.__args__[0].__name__]()

    elif hasattr(typ, "__origin__"):
        if typ.__origin__ is Union:
            # Handle typing.Union
            arg_names = _get_type_names(typ.__args__)
            name = f"Union{arg_names}"
            if name not in CONTRACTS:
                make_union_contract(typ, name)
            contract = CONTRACTS[name]()
        elif typ.__origin__ is list:
            # Handle typing.List object in Python > 3.6
            arg_names = _get_type_names(typ.__args__)
            name = f"List{arg_names}"
            if name not in CONTRACTS:
                make_list_contract(typ, name)
            contract = CONTRACTS[name]()
        elif typ.__origin__ is type:
            # Handle typing.Type object in Python > 3.6
            make_typing_contract(typ)
            contract = CONTRACTS[typ.__args__[0].__name__]()
        else:
            raise ValueError("typing Type but not Union or List")

    # elif typ.__name__ in CONTRACTS:
    #     # if hasattr(typ, '_args'):
    #     #     contract = CONTRACTS[typ.__name__](typ._args)
    #     # else:
    #     contract = CONTRACTS[typ.__name__]()

    if contract is None:
        raise TypeFrameworkError(
            f"Type {typ} (probably custom) has not been added "
            "to the list of acceptable Contracts."
        )
    return contract


def typechecked(func, disable=not ENGAGE_TYPECHECKING):  # noqa (C901)
    """
    Decorator function for annotated functions and class methods

    Parameters
    ----------
    func: callable
        The function or method to decorate

    disable: bool
        Whether or not actually apply typechecking (will default to False)

    Returns
    -------
    result: anything
        The result of the call(s) to func(args, kwargs)

    Raises
    ------
    TypeFrameworkError
        If any of the input arguments or return arguments violate their
        specified annotations
    """
    # Note: This is a complex procedure, which cannot easily be simplified... QA disabled
    # Note: Even if ENGAGE_TYPECHECKING is False, we might want to use this...
    if disable:
        # For speed, if disable is True, just return the function and forget
        # about typechecking. This is for faster use with trusted functions.
        return func

    if not hasattr(func, "__annotations__"):
        # This catches uninstantiated class declarations
        return func

    sig = signature(func)
    ann = func.__annotations__.copy()
    real_ann = ann.copy()

    # Determine the nature of the return signature and separate from inputs
    if "return" in ann:
        has_return = True
        return_type = ann["return"]
        del ann["return"]
    else:
        has_return = False
        return_type = None

    # Get contracts for each of the annotations
    for name, typ in ann.items():
        ann[name] = get_contract(typ)

    if has_return:
        # Handle return argument(s) and get their contracts
        if isinstance(return_type, tuple):
            return_types = []
            for typ in return_type:
                return_types.append(get_contract(typ))
        else:
            return_type = get_contract(return_type)

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        The wrapper function to apply to the func
        """
        bound = sig.bind(*args, **kwargs)
        fails = []

        for arg_name, val in bound.arguments.items():

            if arg_name in ann:
                # If the argument name has an associated annotation, apply
                # contract
                try:
                    if hasattr(sig.parameters[arg_name].annotation, "_args"):
                        annargs = sig.parameters[arg_name].annotation._args
                        ann[arg_name]._verify(val, annargs)
                    else:
                        ann[arg_name]._verify(val)

                except TypeFrameworkError:
                    # Catch failures and process later
                    fails.append([arg_name, val, real_ann[arg_name]])

        if fails:
            handle_fails(func.__module__, func.__name__, fails)

        result = func(*args, **kwargs)

        if not has_return:
            # No return arguments to typecheck
            return result

        # typecheck return argument(s)
        fails = []
        if isinstance(return_type, tuple):
            for i, rtype in enumerate(return_types):
                try:
                    rtype._verify(result[i])
                except TypeFrameworkError:
                    fails.append([i, result[i], real_ann["return"][i]])

            if fails:
                handle_fails(func.__module__, func.__name__, fails, specify="return ")
        else:
            try:
                if hasattr(sig.return_annotation, "_args"):
                    annargs = sig.return_annotation._args
                    return_type._verify(result, annargs)
                else:
                    return_type._verify(result)
            except TypeFrameworkError:

                fails = [[0, result, real_ann["return"]]]
                handle_fails(func.__module__, func.__name__, fails, specify="return ")

        return result

    return wrapper


def handle_fails(module, f_name, fails, specify=""):
    failure = "\n"
    for f in fails:
        failure += (
            f"was expecting {specify}argument '{f[0]}'={f[1]} to be of type '{f[2]}'\n"
        )
    raise TypeFrameworkError(
        f"Error in module: {module}\n" f"function: {f_name}\n" f"{failure}"
    )


if not ENGAGE_TYPECHECKING:
    # Get out the car and go back home to your dynamically typed world.

    class TypeBase:
        """
        The base class when not using the typechecking protocol
        """

        def __init_subclass__(cls):
            """
            Instantiates sub-classes, just assigning None to class parameters
            that have type annotations
            """
            super().__init_subclass__()

            if hasattr(cls, "__annotations__"):
                for name, attribute in cls.__annotations__.items():
                    setattr(cls, name, None)


else:
    # Seatbelt on, welcome to the dynamically type-checked world!

    class TypeBase:
        """
        The base class to use when using the typechecking protocol
        """

        def __init_subclass__(cls):
            """
            Instantiates sub-classes:
                1) decorating the methods in the class with type-checking
                2) applying contracts to the class annotated variables
            """
            super().__init_subclass__()

            # Apply typecheck decorator to the methods in the class
            for name, method in cls.__dict__.items():

                # If the method is a staticmethod or classmethod, we need to
                # 1) un-wrap the underlying func
                # 2) type-wrap
                # 3) static/class-wrap
                if isinstance(method, staticmethod):
                    method = staticmethod(typechecked(method.__func__))

                elif isinstance(method, classmethod):
                    method = classmethod(typechecked(method.__func__))

                # Normal method
                elif callable(method):
                    method = typechecked(method)

                setattr(cls, name, method)

            if hasattr(cls, "__annotations__"):
                # Apply contracts to class annotations
                for name, typ in cls.__annotations__.items():

                    contract = get_contract(typ)
                    contract.__set_name__(cls, name)

                    setattr(cls, name, contract)
                    # I think it we want to enforce defaults, it needs to
                    # happen in __init__...

        @classmethod
        def _verify(cls, value):
            """
            Makes the TypeBase object behave like a Contract, so that we can
            later perform typechecks on its subclasses.
            """
            if not isinstance(value, cls):
                raise TypeFrameworkError


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
