# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
tools for display module
"""

import pprint
from dataclasses import asdict
from typing import Any


class Options:
    """
    The options that are available for displaying objects.
    """

    __slots__ = ("_options",)

    def __init__(self, **kwargs):
        self.modify(**kwargs)

    def __setattr__(self, attr: str, val: Any):
        """
        Set attributes in options dictionary
        """
        if getattr(self, "_options", None) is not None and (
            attr in self._options.__annotations__ or hasattr(self._options, attr)
        ):
            setattr(self._options, attr, val)
        else:
            super().__setattr__(attr, val)

    def __getattribute__(self, attr: str):
        """
        Get attributes or from "_options" dict
        """
        try:
            return super().__getattribute__(attr)
        except AttributeError as ae:
            if attr != "_options":
                try:
                    return getattr(self._options, attr)
                except AttributeError:
                    raise ae from None
            raise

    def modify(self, **kwargs: Any):
        """Modify options"""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def as_dict(self) -> dict[str, Any]:
        """
        Returns the instance as a dictionary.
        """
        return asdict(self._options)

    def __repr__(self) -> str:
        """
        Representation string of the DisplayOptions.
        """
        return f"{type(self).__name__}({pprint.pformat(self.as_dict())}\n)"
