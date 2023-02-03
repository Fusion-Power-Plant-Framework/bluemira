"""
Unit converter object
"""
import inspect
import os

import bluemira.base.file as blm


class DisplayConverter:
    """
    Data converter object, for the purposes of data visualization.

    Takes a parameter with a given unit and converts it to a different
    unit, as defined in the relevant `display_units.JSON` file.

    Parameters
    ----------
    default_display_units: str
        Path to the default `display_units.JSON` file
    desired_display_units: str
        Path to the desired `display_units.JSON` file

    Other Parameters
    ----------------
    """

    def __init__(self, mode):

        # Find relevant `display_units.JSON` files
        self.default_display_units = self.find_default_file()
        self.desired_display_units = self.find_desired_file()

        # Read desired file
        self.display_units = self.read_file(mode)

    def find_default_file(self):
        """
        Look for default `display_units.JSON` file
        """
        project_path = blm.get_bluemira_root()
        default_path = os.path.join("bluemira", "power_cycle")
        default_path = os.path.join(project_path, default_path)
        default_path = os.path.join(default_path, "display_units.json")
        return default_path

    def find_desired_file(self):
        """
        Look for desired `display_units.JSON` file
        """

        # Get path of the top-most script calling the class
        frame = inspect.stack()[-1]
        module = inspect.getmodule(frame[0])
        filename = module.__file__

        # Extract directory name and look for JSON file
        desired_path = os.path.dirname(filename)
        desired_path = os.path.join(desired_path, "display_units.json")
        if not os.path.exists(desired_path):
            desired_path = self.default_display_units

        # If file can't be found, returns default file
        return desired_path

    # def read_desired_file(self):
