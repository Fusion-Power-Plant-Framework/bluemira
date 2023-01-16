from dataclasses import asdict, dataclass
from enum import auto

import numpy as np

from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.codes.interface import CodesSetup, CodesSolver, CodesTask, CodesTeardown
from bluemira.codes.interface import RunMode as BaseRunMode
from bluemira.codes.params import MappedParameterFrame
from bluemira.codes.utilities import ParameterMapping

PRG_NAME = "External Code"
BINARY = ["python", "ext_code_script"]


class RunMode(BaseRunMode):
    """
    RunModes for external code
    """

    RUN = auto()
    READ = auto()
    MOCK = auto()


@dataclass
class ECOpts:
    add_header: bool = False
    number: bool = False

    def to_list(self):
        return [f"--{k.replace('_', '-')}" for k, v in asdict(self).items() if v]


@dataclass
class ECInputs:
    param1: float = np.nan
    param2: float = np.nan


@dataclass
class ECOutputs:
    param1: float = None
    param2: float = None


@dataclass
class ECParameterFrame(MappedParameterFrame):
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
    def mappings(self):
        return self._mappings

    @classmethod
    def from_defaults(cls):
        dd = {}
        for _def in cls._defaults:
            dd = {**dd, **asdict(_def)}
        return super().from_defaults(dd)


class Setup(CodesSetup):

    params: ECParameterFrame

    def __init__(self, params, problem_settings, infile):
        super().__init__(params, PRG_NAME)
        self.problem_settings = problem_settings
        self.infile = infile

    def update_inputs(self):
        self.inputs = ECInputs()
        self.options = ECOpts()
        for thing in (self._get_new_inputs().items(), self.problem_settings.items()):
            for k, v in thing:
                if k in self.options.__annotations__:
                    if v:
                        setattr(self.options, k, v)
                    else:
                        continue
                else:
                    getattr(self.inputs, k)
                    setattr(self.inputs, k, v)

    def run(self):
        self.update_inputs()
        with open(self.infile, "w") as _if:
            _if.write(f"param1  {self.inputs.param1}\n")
            _if.write(f"param2  {self.inputs.param2}\n")
        return self.options.to_list()


class Run(CodesTask):
    def __init__(self, params, infile, outfile, binary=BINARY):
        super().__init__(params, PRG_NAME)
        self.binary = binary
        self.infile = infile
        self.outfile = outfile

    def run(self, options):
        self._run_subprocess([*self.binary, *options, self.infile, self.outfile])


class Teardown(CodesTeardown):
    def __init__(self, params, outfile):
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

    def run(self):
        self._read_file()
        return self.params

    def read(self):
        self._read_file()
        return self.params

    def mock(self):
        self._update_params_with_outputs({"param1": 15})
        return self.params


class Solver(CodesSolver):
    name = PRG_NAME
    params = ECParameterFrame
    setup_cls = Setup
    run_cls = Run
    teardown_cls = Teardown
    run_mode_cls = RunMode

    def __init__(self, params, build_config):

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

    def execute(self, run_mode):
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


def main():
    params = {
        "header": {"value": False, "unit": ""},
        "param1": {"value": 5, "unit": "W"},
    }

    build_config = {
        "problem_settings": {"param2": 10},
        "infile": "infile.txt",
        "outfile": "outfile.txt",
    }

    solver = Solver(params, build_config)
    for mode in ["run", "read", "mock"]:
        print(mode)
        out_params = solver.execute(mode)
        print(out_params)

    solver.modify_mappings({"param2": {"send": False}})
    print(solver.execute("run"))
    build_config["problem_settings"] = {}
    print(solver.execute("run"))
    solver._setup.problem_settings = {}
    print(solver.execute("run"))


main()
