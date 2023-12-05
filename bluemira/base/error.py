# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Bluemira base error class
"""

from textwrap import dedent, fill


class BluemiraError(Exception):
    """
    Base exception class. Sub-class from this for module level Errors.
    """

    def __str__(self) -> str:
        """
        Prettier handling of the Exception strings
        """
        return fill(dedent(self.args[0]))


class BuilderError(BluemiraError):
    """
    Exception class for Builders.
    """


class ComponentError(BluemiraError):
    """
    Exception class for Components.
    """


class LogsError(BluemiraError):
    """
    Exception class for Components.
    """


class ParameterError(BluemiraError):
    """
    Exception class for Parameters.
    """


class DesignError(BluemiraError):
    """
    Exception class for Designs.
    """


class ReactorError(BluemiraError):
    """Exceptions related to :class:`bluemira.base.reactor.Reactor` objects."""


class ReactorConfigError(BluemiraError):
    """
    Exceptions related to
    :class:`bluemira.base.reactor_config.ReactorConfig` objects.
    """
