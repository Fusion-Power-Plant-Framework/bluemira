"""
Base classes for the power cycle model.
"""

'''
import abc
# import matplotlib.pyplot as plt
# from typing import Union


class classproperty(object):
    """
    Class property decorator.
    """

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class PowerCycleABC(abc.ABC):

    @classmethod
    def _issue_error(cls, label):
        """
        Issue error associated with...
        """

        # Retrieve errors of that class
        error_dict = cls._errors

        # Validate error label
        all_labels = error_dict.keys()
        if label in all_labels:

            # Retrieve particular error
            the_error = error_dict[label]

            # Retrieve error attributes
            error_type = the_error.err_type
            error_msg = the_error.err_msg

            # Build raising function
            raise_function = error_type + "Error"
            raise_function = f"""
                raise {raise_function}('{error_msg}')
                """

            exec(raise_function)  # Issue error
        else:

            # Child class name
            class_name = cls.__name__

            # Issue error
            raise ValueError(
                f"""
                Unknown error label for error in class {class_name}.
                """
            )

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------


# ######################################################################
# POWER CYCLE ERROR MESSAGE
# ######################################################################


class PowerCycleError(abc.ABC):
    """
    Abstract base class for handling errors in the Power Cycle module.

    Parameters
    ----------
    err_type: str
        Which type of error is raised when the error instance is called.
    err_msg: str
        Error message displayed by the `raise` command.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES
    # ------------------------------------------------------------------

    # Valid error types
    _valid_types = [
        "Value",
        "Type",
    ]

    # ------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------
    def __init__(self, err_type: str, err_msg: str):

        # Validate inputs
        self.err_type = self._validate_type(err_type)
        self.err_msg = self._validate_msg(err_msg)

    @classmethod
    def _validate_type(cls, err_type):
        """
        Validate `err_type` to be one of the valid types.
        """
        valid_types = cls._valid_types
        if err_type not in valid_types:
            msg_types = PowerCycleUtilities._join_valid_values(valid_types)
            raise ValueError(
                f"""
                The argument given as `err_type` input does not have a
                valid value. Valid values include: {msg_types}.
                """
            )
        return err_type

    @classmethod
    def _validate_msg(cls, err_msg):
        """
        Validate `err_msg` to be a valid message.
        """
        return err_msg

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def _search_keywords(self):
        """
        Find strings preceeded by percent symbols ("%") in the error
        message.
        """

        # Current error message
        err_msg = self.err_msg

        # Split message in whitespaces
        words = err_msg.split()

        # Preallocate output
        keywords = []

        # For each word
        for word in words:

            # If "%" is present, make lowercase and store
            if "%" in word:
                word.lower()
                keywords.append(word)

        # Output keywords
        return keywords
'''
