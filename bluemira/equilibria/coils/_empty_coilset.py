import numpy as np

from bluemira.equilibria.coils._grouping import CoilGroupFieldsMixin, CoilSet


class EmptyCoilSet(CoilSet):
    def __init__(self, *coils, control_names=None):
        if len(coils) > 0:
            raise ValueError("this coilset is not empty")

        attribute_list = dir(self)
        attribute_list.pop(attribute_list.index("__repr__"))

        for method in attribute_list:
            if (
                method.startswith("__")
                or method in ("_quad_boundary")
                or method in CoilGroupFieldsMixin.__slots__
            ):
                continue
            if method in self.__slots__:
                setattr(self, method, self.__return_emptylist)

            if isinstance(getattr(type(self), method, None), property):
                prop = getattr(type(self), method)
                if returntype := prop.fget.__annotations__.get("return", None):
                    if (
                        isinstance(returntype, str)
                        and ("np.ndarray" in returntype or "float" in returntype)
                    ) or (returntype == float or returntype == np.ndarray):
                        setattr(type(self), method, property(self.__return_0))
                    elif (
                        returntype == list
                        or isinstance(returntype, str)
                        and returntype in ("list", "List")
                    ):
                        setattr(type(self), method, property(self.__return_emptylist))
                else:
                    raise NotImplementedError("EmptyCoilset needs a return type")
                continue

            attrib = getattr(self, method)

            if callable(attrib):
                if returntype := attrib.__annotations__.get("return", None):
                    if (
                        isinstance(returntype, str)
                        and ("np.ndarray" in returntype or "float" in returntype)
                    ) or (returntype == float or returntype == np.ndarray):
                        setattr(self, method, self.__return_0)
                else:
                    setattr(self, method, self.__noop)

    def __repr__(self) -> str:
        return "EmptyCoilset"

    def __noop(self, *args, **kwargs):
        pass

    def __return_0(self, *args, **kwargs) -> float:
        return 0

    def __return_emptylist(self, *args, **kwargs) -> list:
        return []
