"""
OptVariablesV2 class
"""
import json
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    TextIO,
    TypedDict,
    Union,
    get_type_hints,
)

import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.display.palettes import BLUEMIRA_PALETTE
from bluemira.utilities.error import OptVariablesError
from bluemira.utilities.tools import json_writer
