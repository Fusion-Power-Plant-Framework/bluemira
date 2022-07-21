"""
Programming tools to make my life easier
"""

# Import necessary packages
# import os
# import sys
# import json
# import numpy as np


class Tools:
    @staticmethod
    def print_header(header=None):
        """
        Print a set of header lines to separate different script runs
        in the terminal.
        """
        # Validate header
        if not header:
            header = "NEW RUN"

        # Build header
        header = " " + header + " "
        header = header.center(72, "=")

        # Print Header
        print("\n\n")
        print(header)
        print("\n")
