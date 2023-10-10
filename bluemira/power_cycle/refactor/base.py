from dataclasses import dataclass
from typing import Any, Dict, Union


@dataclass
class Config:
    name: str


class Descriptor:
    def __set_name__(self, _, name: str):
        """Set the attribute name from a dataclass"""
        self._name = f"_{name}"


class LibraryConfigDescriptor(Descriptor):
    """Config descriptor for use with dataclasses"""

    def __init__(self, *, library_config: Config):
        self.library_config = library_config

    def __get__(self, obj: Any, _) -> Dict[str, Config]:
        """Get the config"""
        return getattr(obj, self._name)

    def __set__(
        self,
        obj: Any,
        value: Dict[str, Union[Config, Dict]],
    ):
        """Setup the config"""
        for k, v in value.items():
            if not isinstance(v, self.library_config):
                value[k] = self.library_config(name=k, **v)

        setattr(obj, self._name, value)
