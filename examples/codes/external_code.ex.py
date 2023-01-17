# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.
"""
External Codes Example
"""
# %%
from dataclasses import asdict, dataclass
from enum import auto
from pprint import pprint
from typing import Dict, List, Union

from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.codes.interface import CodesSetup, CodesSolver, CodesTask, CodesTeardown
from bluemira.codes.interface import RunMode as BaseRunMode
from bluemira.codes.params import MappedParameterFrame
from bluemira.codes.utilities import ParameterMapping

# %% [markdown]
# # External Code Wrapping
#
# This example goes though the minimal steps taken to wrap an external code and
# retrive its outputs
#
# Firstly we define the options available in the code and its name.
# In this example we're wrapping the python script
# [ext_code_script.py](ext_code_script.py).
# Unusually the BINARY global variable is a list instead of a string because there are
# two base commands to run the script from the commandline.
#
# The program has been named "External Code" for simiplicity and is used to help the user
# trace where variables come from. It is a simple script that slightly modifies
# a file provided to it.
#
# There are 3 dataclasses containing the commandline options for the code,
# the input parameters and the output parameters.

# %%
PRG_NAME = "External Code"
BINARY = ["python", "ext_code_script.py"]


@dataclass
class ECOpts:
    """External Code Options"""

    add_header: bool = False
    number: bool = False

    def to_list(self) -> List:
        """Options list"""
        return [f"--{k.replace('_', '-')}" for k, v in asdict(self).items() if v]


@dataclass
class ECInputs:
    """External Code Inputs"""

    param1: float = 6
    param2: float = None


@dataclass
class ECOutputs:
    """External Code Outputs"""

    param1: float = None
    param2: float = None


# %% [markdown]
# ## Linking the code to bluemira
#
# To link an external code to bluemira you need a few bits of machinery
#
# * A `MappedParameterFrame` that links bluemira parameter names and units to the
#   external code
# * A `RunMode` class to specify the possibly running modes
# * A `Solver` that orchistrates the running of the code
# * Some task for the `Solver` to run typically there are three:
#     * A `Setup` task which writes the input file for the code
#     * A `Run` task which runs the code
#     * A `Teardown` task which reads the output file of the code
#
# The `MappedParameterFrame` here gets the defaults and sets the mappings.
# Notice that "param2" is sent to the code but "param1" is not.

# %%
@dataclass
class ECParameterFrame(MappedParameterFrame):
    """External Code ParameterFrame"""

    header: Parameter[bool]
    line_number: Parameter[bool]
    param1: Parameter[float]
    param2: Parameter[float]

    _defaults = (ECOpts(), ECInputs())

    _mappings = {
        "header": ParameterMapping("add_header", send=True, recv=False),
        "line_number": ParameterMapping("number", send=True, recv=False),
        "param1": ParameterMapping("param1", send=False, recv=True, unit="MW"),
        "param2": ParameterMapping("param2", send=True, recv=True, unit="GW"),
    }

    @property
    def mappings(self) -> Dict:
        """Code Mappings"""
        return self._mappings

    @classmethod
    def from_defaults(cls) -> MappedParameterFrame:
        """Setup from defaults"""
        dd = {}
        for _def in cls._defaults:
            dd = {**dd, **asdict(_def)}
        return super().from_defaults(dd)


# %%
class RunMode(BaseRunMode):
    """
    RunModes for external code
    """

    RUN = auto()
    READ = auto()
    MOCK = auto()


# %% [markdown]
# The `Setup` class pulls over the inputs from bluemira as described by the
# mapping and creates the input file. The output of the run method returns
# the commandline options list.

# %%
class Setup(CodesSetup):
    """Setup task"""

    params: ECParameterFrame

    def __init__(self, params: ParameterFrame, problem_settings: Dict, infile: str):
        super().__init__(params, PRG_NAME)
        self.problem_settings = problem_settings
        self.infile = infile

    def update_inputs(self) -> Dict:
        """Update inputs from bluemira"""
        self.inputs = ECInputs()
        self.options = ECOpts()
        inp = self._get_new_inputs()
        for thing in (self.problem_settings.items(), inp.items()):
            for k, v in thing:
                if k in self.options.__annotations__:
                    if v:
                        setattr(self.options, k, v)
                    else:
                        continue
                elif v is not None:
                    getattr(self.inputs, k)
                    setattr(self.inputs, k, v)

        return {k: getattr(self.inputs, k) for k, v in inp.items() if v}

    def run(self) -> List:
        """Run mode"""
        inp = self.update_inputs()
        with open(self.infile, "w") as _if:
            for k, v in inp.items():
                _if.write(f"{k}  {v}\n")
        return self.options.to_list()


# %% [markdown]
# `Run` simply runs the code in a subprocess with the given options

# %%
class Run(CodesTask):
    """Run task"""

    def __init__(
        self, params: ParameterFrame, infile: str, outfile: str, binary: List = BINARY
    ):
        super().__init__(params, PRG_NAME)
        self.binary = binary
        self.infile = infile
        self.outfile = outfile

    def run(self, options: List):
        """Run mode"""
        self._run_subprocess([*self.binary, *options, self.infile, self.outfile])


# %% [markdown]
# `Teardown` reads in a given output file or in the case of mock returns a known
# value sending the new parameter values back to the `ParameterFrame`

# %%
class Teardown(CodesTeardown):
    """Teardown task"""

    def __init__(self, params: ParameterFrame, outfile: str):
        super().__init__(params, PRG_NAME)
        self.outfile = outfile

    def _read_file(self):
        out_params = {}
        with open(self.outfile, "r") as of:
            for line in of:
                if line.startswith("#"):
                    pass
                if line.startswith(" "):
                    k, v = line.split()
                    out_params[k] = float(v)
        self._update_params_with_outputs(out_params)

    def run(self) -> ParameterFrame:
        """Run mode"""
        self._read_file()
        return self.params

    def read(self) -> ParameterFrame:
        """Read mode"""
        self._read_file()
        return self.params

    def mock(self) -> ParameterFrame:
        """Mock mode"""
        self._update_params_with_outputs({"param1": 15})
        return self.params


# %% [markdown]
# `Solver` combines the three tasks into one object for execution.
# The execute method has been overridden here for our usecase and returns
# the `ParameterFrame`.

# %%
class Solver(CodesSolver):
    """The External Code Solver."""

    name = PRG_NAME
    params = ECParameterFrame
    setup_cls = Setup
    run_cls = Run
    teardown_cls = Teardown
    run_mode_cls = RunMode

    def __init__(self, params: Union[ParameterFrame, Dict], build_config: Dict):

        self.params = ECParameterFrame.from_defaults()

        if isinstance(params, ParameterFrame):
            self.params.update_from_frame(params)
        else:
            try:
                self.params.update_from_dict(params)
            except TypeError:
                self.params.update_values(params)
        self._setup = self.setup_cls(
            self.params, build_config.get("problem_settings", {}), build_config["infile"]
        )
        self._run = self.run_cls(
            self.params,
            build_config["infile"],
            build_config["outfile"],
            build_config.get("binary", BINARY),
        )
        self._teardown = self.teardown_cls(self.params, build_config["outfile"])

    def execute(self, run_mode: Union[str, RunMode]) -> ParameterFrame:
        """Execute the solver"""
        if isinstance(run_mode, str):
            run_mode = self.run_mode_cls.from_string(run_mode)
        result = None
        if setup := self._get_execution_method(self._setup, run_mode):
            result = setup()
        if run := self._get_execution_method(self._run, run_mode):
            run(result)
        if teardown := self._get_execution_method(self._teardown, run_mode):
            result = teardown()
        return result


# %% [markdown]
# ### Using the solver
#
# To run the solver you just need to provide the parameters and the configuration
# to initialse the object.
# Be aware `problem_settings` should be used sparingly for options that won't
# change within the rerunning of the solver,
# it has the same effect as modifying the default.
# Also note the units of the parameter values returned back have been updated

# %%
params = {
    "header": {"value": False, "unit": "", "source": "here"},
    "param1": {"value": 5, "unit": "W"},
}

build_config = {
    "problem_settings": {"param2": 10},
    "infile": "infile.txt",
    "outfile": "outfile.txt",
}

solver = Solver(params, build_config)

# Running in all the different modes
for mode in ["run", "read", "mock"]:
    print(mode)
    out_params = solver.execute(mode)
    print(out_params)

# %%
solver.modify_mappings({"param2": {"send": False}})
# Change from the previous run method because mapping
# has changed
print(solver.execute("run"))

# %%
# problem_settings param2 overridden
solver.modify_mappings({"param2": {"send": True}})
solver.params.param2.value = 5
print(solver.execute("run"))

# %%
# param 2 not sent
solver.modify_mappings({"param2": {"send": False}})
solver.params.param2.value = None
print(solver.execute("run"))

# %%
# problem_setting param2 sent
solver.modify_mappings({"param2": {"send": True}})
print(solver.execute("run"))

# %%
# param 1 sent default used
solver.modify_mappings({"param1": {"send": True}})
solver.params.param2.set_value(5, "param1 sent with default value")
print(solver.execute("run"))

# %%
# param 1 sent
solver.params.param1.value = 5
solver.params.param2.value = 5
print(solver.execute("run"))

# %%
for param in solver.params:
    print(param.name)
    pprint(param.history())
