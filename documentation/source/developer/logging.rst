Logging in Bluemira
-------------------

To organise and have granularity in text (and potentially graphical) outputs
we use some wrappers around python's built-in logging module.
This allows us to store the output in a log file whilst also being precise about
what information is displayed in standard out.

What does this mean for a developer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a few actions that should be taken:

* All printing or stdout calls should go through one of our logging level print functions

    .. code-block:: python

        from bluemira.base.look_and_feel import (
            bluemira_critical,
            bluemira_debug,
            bluemira_error,
            bluemira_print,
            bluemira_warn,
        )

        bluemira_print('my message')

* The current logging level controls the amount of output to the screen not to the
  logfile which records everything that is logged.

* Logging level can be controlled with the :py:func:`set_log_level` function

    .. code-block:: python

       from bluemira.base.logs import get_log_level, set_log_level

       get_log_level()  # INFO
       set_log_level('DEBUG')
       get_log_level()  # DEBUG

* Verbosity and/or debugging function input variables should not be used but the output
  should be defined by the current log level

    .. code-block:: python

        from bluemira.base.logs import get_log_level

        # Possibility if verbosity wants to be explicit
        def my_function(verbosity=get_log_level(as_str=False) < 2):
            if verbosity:
                # Do stuff


        # A cleaner but less customisable version
        def my_function():
            loglevel = get_log_level()
            if loglevel == "DEBUG":
                # Do stuff

* For updated text (e.g. progress bars) there are a few functions that do not print newlines


    .. code-block:: python

        from bluemira.base.look_and_feel import (
            bluemira_debug_flush,
            bluemira_print_flush,
        )

        bluemira_print_flush('my message')
        bluemira_print_flush('my other message')  # overwrites the first message

* If formatted output is not desired, there are error and print functions that direct to the logger
  as needed with no other modifications. This is used internally for external code outputs.

    .. code-block:: python

        from bluemira.base.look_and_feel import (
            bluemira_print_clean,
            bluemira_error_clean,
        )

        bluemira_print_clean('my message')  # Not coloured or formatted
        bluemira_error_clean('my other message')  # Coloured but not formatted
