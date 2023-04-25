# %%
# COPYRIGHT PLACEHOLDER

"""
Path magic to import 'kits_for_examples' from parent directory.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def successfull_import():
    """
    Print success message.
    """
    print("Imported 'kits_for_examples' successfully.")


def failed_import():
    """
    Print failure message.
    """
    print("Failed to import 'kits_for_examples'.")
