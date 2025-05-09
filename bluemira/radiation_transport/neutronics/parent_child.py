# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Module to contain a class that accepts uplinks (i.e. links back to the parent)."""

from bluemira.radiation_transport.neutronics.error import (
    MissingParentError,
    ReadOnlyAttributeError,
)


class ParentLinkable:
    """An item that can be linked to its parent. For example,
    type(obj.child[0]) = ParentLinkable
    obj.child[0].parent.child[0] == obj.child[0]

    At the moment of instiation, .parent is set to None. Therefore the following logic
    is needed to make it .parent set-able, but once written, cannot be over-written.
    """

    @property
    def parent(self):
        """Returns the parent if set. Otherwise throw an error.

        Raises
        ------
        MissingParentError
            Thrown if no parent is set.
        """
        if self._parent:
            return self._parent
        raise MissingParentError(
            f"{self} does not belong in a {self._allowed_parent_class} yet!"
        )

    @parent.setter
    def parent(self, _parent):
        """
        Set the parent if it hasn't been set. Otherwise, throw an error.

        Raises
        ------
        TypeError
            Incorrect parent type used.
        ReadOnlyAttributeError
            Forbid changing the parent once set.
        """
        if not isinstance(_parent, self._allowed_parent_class):
            raise TypeError(
                f"The parent of a {self.__class__} must be a "
                f"{self._allowed_parent_class}."
            )
        if self._parent:
            raise ReadOnlyAttributeError(
                f"We do not allow changing the ownership of {self.__class__} to a new "
                f"parent. If you are trying to instantialize a new instance of "
                f"{self._allowed_parent_class} for {self}, same please use "
                "link_children=False."
            )
        self._parent = _parent
