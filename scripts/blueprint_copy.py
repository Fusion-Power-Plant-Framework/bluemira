"""
Script for copying files between the legacy BLUEPRINT repo and this bluemira repo while
the two exist in parallel during the migration. It performs a series of replacements for
files that have been changed in the legacy BLUEPRINT modules in order to ensure that
bluemira changes are not lost, while maintaining the changes made in BLUEPRINT.

Note that in some cases changes must be applied manually to files that have been migrated
over to bluemira to ensure that functionality is not lost.
"""

import pathlib
import shutil
import subprocess  # noqa S404 - subprocess used to make git calls

from bluemira.base.file import get_bluemira_root

old_header = """# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>."""

new_header = """# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>."""

replacements = {
    old_header: new_header,
    "from BLUEPRINT.base.lookandfeel import get_BP_path, bprintflush, bpwarn": """from BLUEPRINT.base.file import get_BP_path
from bluemira.base.look_and_feel import bluemira_warn, bluemira_print_flush""",
    "bprintflush": "bluemira_print_flush",
    "bprint": "bluemira_print",
    "bpwarn": "bluemira_warn",
    "from .lookandfeel": "from bluemira.base.look_and_feel",
    "from .file import make_BP_path, get_files_by_ext, FileManager, SUB_DIRS": """from .file import make_BP_path, FileManager, SUB_DIRS
from bluemira.base.file import get_files_by_ext""",
    '"tests"': '"tests/BLUEPRINT"',
    " banner": " print_banner",
    "from BLUEPRINT.base.lookandfeel import plot_defaults, KEY_TO_PLOT": """from BLUEPRINT.base.lookandfeel import KEY_TO_PLOT
from bluemira.base.look_and_feel import plot_defaults""",
    """from BLUEPRINT.base import (
    ReactorSystem,
    BLUE,
    bluemira_print,
    print_banner,
    bluemira_warn,
    get_files_by_ext,
)""": """from BLUEPRINT.base import (
    ReactorSystem,
    BLUE,
)
from bluemira.base.file import get_files_by_ext
from bluemira.base.look_and_feel import bluemira_warn, bluemira_print, print_banner
from bluemira.base.error import BluemiraError""",
    "BLUEPRINT.base.lookandfeel": "bluemira.base.look_and_feel",
    "from bluemira.base.look_and_feel import KEY_TO_PLOT": "from BLUEPRINT.base.lookandfeel import KEY_TO_PLOT",
    "from BLUEPRINT.base.error import BLUEPRINTError, ": "from BLUEPRINT.base.error import ",
    "BLUEPRINT.base.constants": "bluemira.base.constants",
    "BLUEPRINT.base.logs": "bluemira.base.logs",
    '"data"': '"data/BLUEPRINT"',
    '"generated_data"': '"generated_data/BLUEPRINT"',
    '"!BP_ROOT!/data"': '"!BP_ROOT!/data/BLUEPRINT"',
    "BLUEPRINTError": "BluemiraError",
    "BClock": "BluemiraClock",
    "from BLUEPRINT.base.file import get_files_by_ext": "from bluemira.base.file import get_files_by_ext",
    ": [float, float, float], ": ", ",
    """from natsort import natsorted
""": "",
    "natsorted": "sorted",
}

root = get_bluemira_root()
old_code_dir = pathlib.Path(root, "..", "BLUEPRINT", "BLUEPRINT")
new_code_dir = pathlib.Path(root, "BLUEPRINT")

for path in old_code_dir.rglob("*.py"):
    path = str(path)
    target_path = path.replace(str(old_code_dir), str(new_code_dir))
    shutil.copy(str(path), target_path)

for path in new_code_dir.rglob("*.py"):
    content = path.read_text()
    if "baseclass.py" in str(path):
        content = content.replace(
            "from BLUEPRINT.base.parameter import ParameterFrame",
            """from BLUEPRINT.base.parameter import ParameterFrame
from bluemira.base.error import BluemiraError""",
        )
    for old, new in replacements.items():
        content = content.replace(old, new)
    path.write_text(content)

    if (
        "lookandfeel.py" in str(path)
        or "base/error.py" in str(path)
        or "base/file.py" in str(path)
        or "equilibria/profiles.py" in str(path)
    ):
        # Use a subprocess call to reset files via git that shouldn't be changed or
        # reintroduced. Note that this functionality doesn't seem to be available via the
        # git Python API.
        proc = subprocess.Popen(["git", "checkout", "--", str(path)])  # noqa S603, S607
        proc.wait()

    if "base/constants.py" in str(path) or "base/logs.py" in str(path):
        path.unlink()

# Use a subprocess call to run black. Note this doesn't seem to be (easily) doable via
# the black API.
proc = subprocess.Popen(["black", root])  # noqa S603, S607
proc.wait()


# Handle documentation
old_docs_dir = pathlib.Path(root, "..", "BLUEPRINT", "documentation")
new_docs_dir = pathlib.Path(root, "documentation", "BLUEPRINT")

for path in old_docs_dir.rglob("*.rst"):
    path = str(path)
    target_path = path.replace(str(old_docs_dir), str(new_docs_dir))
    shutil.copy(str(path), target_path)


# Handle tests
old_tests_dir = pathlib.Path(root, "..", "BLUEPRINT", "tests")
new_tests_dir = pathlib.Path(root, "tests", "BLUEPRINT")

for path in old_tests_dir.rglob("*.py"):
    path = str(path)
    target_path = path.replace(str(old_tests_dir), str(new_tests_dir))
    shutil.copy(str(path), target_path)

for path in old_tests_dir.rglob("*.pkl"):
    path = str(path)
    target_path = path.replace(str(old_tests_dir), str(new_tests_dir))
    shutil.copy(str(path), target_path)

for path in old_tests_dir.rglob("*.png"):
    path = str(path)
    target_path = path.replace(str(old_tests_dir), str(new_tests_dir))
    shutil.copy(str(path), target_path)

for path in old_tests_dir.rglob("*.json"):
    path = str(path)
    target_path = path.replace(str(old_tests_dir), str(new_tests_dir))
    shutil.copy(str(path), target_path)

for path in old_tests_dir.rglob("*.csv"):
    path = str(path)
    target_path = path.replace(str(old_tests_dir), str(new_tests_dir))
    shutil.copy(str(path), target_path)

replacements = {
    old_header: new_header,
    "tests.test_reactor": "tests.BLUEPRINT.test_reactor",
    "tests/test_data": "tests/BLUEPRINT/test_data",
    "tests/test_generated_data": "tests/BLUEPRINT/test_generated_data",
    'tempdir, "test_data"': 'tempdir, "BLUEPRINT", "test_data"',
    'tempdir, "test_generated_data"': 'tempdir, "BLUEPRINT", "test_generated_data"',
    """\"tests\",
        \"test_data\"""": """\"tests\",
        \"BLUEPRINT\",
        \"test_data\"""",
    """\"tests\",
        \"test_generated_data\"""": """\"tests\",
        \"BLUEPRINT\",
        \"test_generated_data\"""",
    '"tests", "base"': '"tests", "BLUEPRINT", "base"',
    '"base", subfolder="tests"': '"BLUEPRINT/base", subfolder="tests"',
    '"beams", subfolder="tests"': '"BLUEPRINT/beams", subfolder="tests"',
    '"cad", subfolder="tests"': '"BLUEPRINT/cad", subfolder="tests"',
    '"cad/test_data", subfolder="tests"': '"BLUEPRINT/cad/test_data", subfolder="tests"',
    '"tests", "cli"': '"tests", "BLUEPRINT", "cli"',
    '"equilibria/test_data", subfolder="tests"': '"BLUEPRINT/equilibria/test_data", subfolder="tests"',
    '"tests/equilibria/test_data/': '"tests/BLUEPRINT/equilibria/test_data/',
    'path="equilibria", subfolder="data"': 'path="equilibria", subfolder="data/BLUEPRINT"',
    '"equilibria/test_data", "tests"': '"equilibria/test_data", "tests/BLUEPRINT"',
    '"eqdsk", subfolder="data"': '"eqdsk", subfolder="data/BLUEPRINT"',
    '"fuelcycle/blanket_fw_T_retention", subfolder="data"': '"fuelcycle/blanket_fw_T_retention", subfolder="data/BLUEPRINT"',
    '"geometry", subfolder="tests"': '"geometry/BLUEPRINT", subfolder="tests"',
    '"geometry", "tests"': '"BLUEPRINT/geometry", "tests"',
    '"geometry/test_data", subfolder="tests"': '"BLUEPRINT/geometry/test_data", subfolder="tests"',
    '"geometry/test_generated_data", subfolder="tests"': '"BLUEPRINT/geometry/test_generated_data", subfolder="tests"',
    '"Geometry", subfolder="data"': '"Geometry", subfolder="data/BLUEPRINT"',
    'os.sep.join(["Geometry"]), subfolder="data"': 'os.sep.join(["Geometry"]), subfolder="data/BLUEPRINT"',
    '"magnetostatics/test_data", subfolder="tests"': '"BLUEPRINT/magnetostatics/test_data", subfolder="tests"',
    '"nova/test_data", subfolder="tests"': '"BLUEPRINT/nova/test_data", subfolder="tests"',
    '"utilities/test_data", subfolder="tests"': '"BLUEPRINT/utilities/test_data", subfolder="tests"',
    "tests.base.": "tests.BLUEPRINT.base.",
    "tests.cad.": "tests.BLUEPRINT.cad.",
    "tests.equilibria.": "tests.BLUEPRINT.equilibria.",
    "tests.magnetostatics.": "tests.BLUEPRINT.magnetostatics.",
    "tests.systems.": "tests.BLUEPRINT.systems.",
    "BLUEPRINT.base.error import BLUEPRINTError": "bluemira.base.error import BluemiraError",
    '"test_data/reactors/SMOKE-TEST", subfolder="tests"': '"BLUEPRINT/test_data/reactors/SMOKE-TEST", subfolder="tests"',
    "BLUEPRINTError": "BluemiraError",
    "BLUEPRINT.base.constants": "bluemira.base.constants",
    "BLUEPRINT.base.logs": "bluemira.base.logs",
    "from BLUEPRINT.base.lookandfeel import plot_defaults": "from bluemira.base.look_and_feel import plot_defaults",
    "from BLUEPRINT.base.lookandfeel import bprint": "from bluemira.base.look_and_feel import bluemira_print",
    "from BLUEPRINT.base.file import get_BP_path, get_files_by_ext": """from BLUEPRINT.base.file import get_BP_path
from bluemira.base.file import get_files_by_ext""",
    'with patch("BLUEPRINT.equilibria.positioner.bpwarn") as bpwarn:': 'with patch("BLUEPRINT.equilibria.positioner.bluemira_warn") as bluemira_warn:',
    "bpwarn": "bluemira_warn",
    "bprint": "bluemira_print",
    '"syscodes/test_data"': '"BLUEPRINT/syscodes/test_data"',
}

for path in new_tests_dir.rglob("*.py"):
    content = path.read_text()
    for old, new in replacements.items():
        content = content.replace(old, new)
    path.write_text(content)

    if "baseline_systems_pickles.py" in str(path) or "base/test_lookandfeel.py" in str(
        path
    ):
        # Use a subprocess call to reset files via git that shouldn't be changed or
        # reintroduced. Note that this functionality doesn't seem to be available via the
        # git Python API.
        proc = subprocess.Popen(["git", "checkout", "--", str(path)])  # noqa S603, S607
        proc.wait()

    if "base/test_logs.py" in str(path):
        path.unlink()

replacements = {
    "!BP_ROOT!/tests/test_data": "!BP_ROOT!/tests/BLUEPRINT/test_data",
    "!BP_ROOT!/tests/test_generated_data": "!BP_ROOT!/tests/BLUEPRINT/test_generated_data",
}

for path in new_tests_dir.rglob("*.json"):
    content = path.read_text()
    for old, new in replacements.items():
        content = content.replace(old, new)
    path.write_text(content)

# Use a subprocess call to run black. Note this doesn't seem to be (easily) doable via
# the black API.
proc = subprocess.Popen(["black", root])  # noqa S603, S607
proc.wait()


# Handle examples

old_examples_dir = pathlib.Path(root, "..", "BLUEPRINT", "examples")
new_examples_dir = pathlib.Path(root, "examples", "BLUEPRINT")

replacements = {
    old_header: new_header,
    '"data"': '"data/BLUEPRINT"',
    '"generated_data"': '"generated_data/BLUEPRINT"',
    '"!BP_ROOT!/data"': '"!BP_ROOT!/data/BLUEPRINT"',
    '"!BP_ROOT!/generated_data"': '"!BP_ROOT!/generated_data/BLUEPRINT"',
    "from BLUEPRINT.base.lookandfeel import plot_defaults": "from bluemira.base.look_and_feel import plot_defaults",
    '"cad/test_data", subfolder="tests"': '"BLUEPRINT/cad/test_data", subfolder="tests"',
    '\\"cad/test_data\\", subfolder=\\"tests\\"': '\\"BLUEPRINT/cad/test_data\\", subfolder=\\"tests\\"',
    "examples/systems": "examples/BLUEPRINT/systems",
    "from BLUEPRINT.base.lookandfeel import bprint": "from bluemira.base.look_and_feel import bluemira_print",
    "bprint": "bluemira_print",
    "plt.ion()\nplt.show()\n": "",
    "plt.ion()\n": "",
    """# %%[markdown]
# First wall shape, hit points and heat flux values""": """plt.show()

# %%[markdown]
# First wall shape, hit points and heat flux values""",
    """# %%[markdown]
# Heat flux values against poloidal location""": """plt.show()

# %%[markdown]
# Heat flux values against poloidal location""",
}

for path in old_examples_dir.rglob("*.py"):
    path = str(path)
    target_path = path.replace(str(old_examples_dir), str(new_examples_dir))
    shutil.copy(str(path), target_path)

for path in old_examples_dir.rglob("*.ipynb"):
    path = str(path)
    target_path = path.replace(str(old_examples_dir), str(new_examples_dir))
    shutil.copy(str(path), target_path)

for path in new_examples_dir.rglob("*.py"):
    content = path.read_text()
    for old, new in replacements.items():
        content = content.replace(old, new)
    path.write_text(content)

    if (
        "ST.py" in str(path)
        or "openmoc" in str(path)
        or "divertor_silhouette" in str(path)
    ):
        # Use a subprocess call to reset files via git that shouldn't be changed or
        # reintroduced. Note that this functionality doesn't seem to be available via the
        # git Python API.
        proc = subprocess.Popen(["git", "checkout", "--", str(path)])  # noqa S603, S607
        proc.wait()

for path in new_examples_dir.rglob("*.ipynb"):
    content = path.read_text()
    for old, new in replacements.items():
        content = content.replace(old, new)
    path.write_text(content)

    if "divertor_silhouette" in str(path):
        # Use a subprocess call to reset files via git that shouldn't be changed or
        # reintroduced. Note that this functionality doesn't seem to be available via the
        # git Python API.
        proc = subprocess.Popen(["git", "checkout", "--", str(path)])  # noqa S603, S607
        proc.wait()

# Use a subprocess call to run black. Note this doesn't seem to be (easily) doable via
# the black API.
proc = subprocess.Popen(["black", root])  # noqa S603, S607
proc.wait()
