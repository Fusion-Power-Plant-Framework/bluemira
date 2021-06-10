from ._version import get_versions

__version__ = get_versions()["version"]


from . import const
from . import geo
from . import plotting
from . import meshing
from . import core
from . import algebra
from . import emag
from . import dolfinSolver
from . import equilibrium
from . import machine
from . import femhelper
from . import Utils
from . import critical
