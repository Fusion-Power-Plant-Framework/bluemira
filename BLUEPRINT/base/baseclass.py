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

"""
Home of the BLUEPRINT base class for reactor system objects
"""
from copy import deepcopy
import pickle  # noqa (S403)
from typing import Union, Type

from BLUEPRINT.base.typebase import TypeBase
from BLUEPRINT.base.parameter import ParameterFrame
from BLUEPRINT.base.error import BLUEPRINTError, SystemsError


class ReactorSystem(TypeBase):
    """
    The standard BLUEPRINT system class.

    Specify inputs directly in the body under the definition of the class.
    Variable inputs are handled with descriptors. You must specify the type of
    expected input, and any constraints/limits if necessary. All type-checking
    is handled automatically.

    All inherited classes are granted the following attributes (this is
    implemented in the metaclass, as otherwise they would be public
    attributes and would overwrite each other).

    Attributes
    ----------
    geom: dict
        The geometry storage dictionary
    requirements: dict
        The requirements storage dictionary
    params: ParameterFrame
        The ReactorSystem ParameterFrame
    """

    CADConstructor = NotImplemented
    __available_classes = {}
    _subsystem_base_classes = {}

    def __init_subclass__(cls):
        """
        Initialise a reactor sub-class
        """
        super().__init_subclass__()

        # Get the default_params (or those of the parent if not overridden)
        default_params = getattr(cls, "default_params", [])
        if isinstance(default_params, ParameterFrame):
            default_params = default_params.to_records()
        cls.default_params = ParameterFrame(default_params)

        # Build a registry of unique ReactorSystem classes for access by name.
        if cls.__name__ not in cls.__available_classes:
            cls.__available_classes[cls.__name__] = cls
        else:
            raise BLUEPRINTError(
                f"A class with name {cls.__name__} is already defined in BLUEPRINT as a "
                f"ReactorSystem : {cls.__available_classes[cls.__name__]} from "
                f"{cls.__available_classes[cls.__name__].__module__}. You tried to "
                f"specify {cls} from {cls.__module__}."
            )

        # Get the subsystem base classes from the parent class
        cls._subsystem_base_classes = {}
        for parent_class in cls.__bases__:
            if issubclass(parent_class, ReactorSystem):
                cls._subsystem_base_classes.update(parent_class._subsystem_base_classes)

        # Add any new subsystem base classes (or perform any overrides)
        if hasattr(cls, "__annotations__"):
            for name, ty in cls.__annotations__.items():
                if hasattr(ty, "__origin__") and (
                    ty.__origin__ is Type or ty.__origin__ is type
                ):
                    # Only handle single base classes for now.
                    if len(ty.__args__) == 1 and issubclass(
                        ty.__args__[0], ReactorSystem
                    ):
                        cls._subsystem_base_classes[name] = ty.__args__[0]

    def __new__(cls, *args, **kwargs):
        """
        Add some default attributes to the sub-classes of the ReactorSystem
        """
        self = super().__new__(cls)
        self.geom = {}
        self.requirements = {}
        return self

    @classmethod
    def get_class(cls, name: str):
        """
        Get the class with the provided name.

        The requested class must be either the class itself or one of its subclasses.

        Parameters
        ----------
        name: str
            The class name.

        Returns
        -------
        the_class: ReactorSystem
            The class with the specified name.
        """
        if name not in cls.__available_classes:
            raise BLUEPRINTError(
                f"{name} is not known as a BLUEPRINT ReactorSystem. Either ensure that "
                "the class inherits from a ReactorSystem or check that you have "
                "imported the required module."
            )

        the_class = cls.__available_classes[name]
        if issubclass(the_class, cls):
            return the_class
        else:
            raise BLUEPRINTError(
                f"Unable to find {name} as a subclass of {cls.__name__}"
            )

    def _generate_subsystem_classes(self, config):
        """
        Generate a dictionary of classes to be used in the build, and validate them.
        """
        self._subsystem_classes = {}
        for name, ty in self._subsystem_base_classes.items():
            config_key = f"{name.lower()}_class_name"
            class_name = config.get(config_key, ty.__name__)
            self._subsystem_classes[name] = ty.get_class(class_name)

    def get_subsystem_class(self, key):
        """
        Get the subsystem class corresponding to the key.

        Parameters
        ----------
        key: str
            The subsystem key.

        Returns
        -------
        system_class: Type[ReactorSystem]
            The subsystem class corresponding to the provided key.
        """
        try:
            return self._subsystem_classes[key]
        except KeyError:
            raise SystemsError(f"Unknown subsystem key {key} requested.")

    def add_parameter(
        self,
        var: str,
        name: str,
        value,  #: Union[str, int, float, np.float64],
        unit: Union[str, None],
        descr: Union[str, None],
        source: Union[str, None],
        mapping=None,
    ):
        """
        Add a parameter to the ReactorSystem ParameterFrame

        Parameters
        ----------
        var: str
            The short parameter name
        name: str
            The long parameter name
        value: float
            The value of the parameter
        unit: str
            The unit of the parameter
        descr: str
            The long description of the parameter
        source: str
            The source (reference and/or code) of the parameter
        mapping: Union[Dict[str, ParameterMapping], None]
            The names used for this parameter in external software, and whether
            that parameter should be written to and/or read from the external tool
        """
        self.params.add_parameter(var, name, value, unit, descr, source, mapping)

    def add_parameters(self, record_list):
        """
        Handles lists for new Parameters or dicts for existing keywords only
        """
        if isinstance(record_list, list):
            self.params.add_parameters(record_list)
        elif isinstance(record_list, dict):
            self.params.update_kw_parameters(record_list)

    def build_CAD(self):
        """
        Builds the CAD model for the ReactorSystem
        """
        cad = self.CADConstructor(self)
        self._seg_props = cad.get_properties()
        return cad

    def show_CAD(self):
        """
        Shows the CAD model for the ReactorSystem
        """
        from BLUEPRINT.cad.model import CADModel  # Circular referencing

        model = CADModel(self.params.n_TF)
        model.add_part(self.build_CAD())
        # for debugging
        self._CADModel = model
        model.display()

    @property
    def xy_plot_loop_names(self):
        """
        The names of the loops to be plotted in the X-Y mid-plane.

        Returns
        -------
        List[str]
            The names of the loops to be plotted in the X-Y mid-plane.
        """
        return []

    def _generate_xy_plot_loops(self):
        """
        Generate the loops to be plotted in the X-Y mid-plane.

        Returns
        -------
        List[Loop]
            The loops to be plotted in the X-Y mid-plane.
        """
        return [self.geom[key] for key in self.xy_plot_loop_names]

    def plot_xy(self, ax=None, **kwargs):
        """
        Generate a plot in the X-Y mid-plane.

        Parameters
        ----------
        ax : Axes, optional
            The Axes to plot onto, by default None.
            If None then the current Axes will be used.
        """
        self._plotter.plot_xy(self._generate_xy_plot_loops(), ax=ax, **kwargs)

    @property
    def xz_plot_loop_names(self):
        """
        The names of the loops to be plotted in the X-Z plane.

        Returns
        -------
        List[str]
            The names of the loops to be plotted in the X-Z plane.
        """
        return []

    def _generate_xz_plot_loops(self):
        """
        Generate the loops to be plotted in the X-Z plane.

        Returns
        -------
        List[Loop]
            The loops to be plotted in the X-Z plane.
        """
        return [self.geom[key] for key in self.xz_plot_loop_names]

    def plot_xz(self, ax=None, **kwargs):
        """
        Generate a plot in the X-Z plane.

        Parameters
        ----------
        ax: Axes, optional
            The Axes to plot onto, by default None.
            If None then the current Axes will be used.
        """
        self._plotter.plot_xz(self._generate_xz_plot_loops(), ax=ax, **kwargs)

    def save(self, path):
        """
        Save a ReactorSystem object to a pickle file.

        Parameters
        ----------
        path: str
            Full path with filename
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        """
        Load a ReactorSystem object from a pickle file.

        Parameters
        ----------
        path: str
            Full path with filename
        """
        with open(path, "rb") as file:
            return pickle.load(file)  # noqa (S301)

    def __getstate__(self):
        """
        Pickling utility.

        Pickles the params and config as records (lists of values).
        """
        d = dict(self.__dict__)
        if isinstance(getattr(self, "params", None), ParameterFrame):
            d.pop("params", None)
            d["__parameter_frame__"] = self.params.to_records()
        if isinstance(getattr(self, "config", None), ParameterFrame):
            d.pop("config", None)
            d["__config_frame__"] = self.config.to_records()
        return d

    def __setstate__(self, state):
        """
        Un-Pickling utility.

        Un-pickles the params and config from records (lists of values).
        """
        params = state.pop("__parameter_frame__", None)
        config = state.pop("__config_frame__", None)
        self.__dict__ = state
        if params is not None:
            self.__dict__["params"] = ParameterFrame(params)
        if config is not None:
            self.__dict__["config"] = ParameterFrame(config)

    def copy(self):
        """
        Provides a deep copy of the ReactorSystem

        Returns
        -------
        copy: ReactorSystem
            The copy of the ReactorSystem
        """
        return deepcopy(self)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
