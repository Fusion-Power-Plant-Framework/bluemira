import argparse
import inspect
from copy import deepcopy
from pathlib import Path

from bluemira.base.parameter_frame._parameter import ParamDictT
from bluemira.utilities.tools import get_module, json_writer


def def_param():
    dp = ParamDictT.__annotations__
    del dp["name"]
    for k in dp:
        if k != "value":
            dp[k] = "str"
    return dp


DEFAULT_PARAM = def_param()


def add_to_dict(vv, out_dict, params):
    for param, param_type in vv.__annotations__.items():
        dv = deepcopy(DEFAULT_PARAM)
        dv["value"] = str(param_type).split("[")[-1][:-1]
        out_dict[param] = dv
        params[param] = param_type


def create_parameterframe(params, name=None, header=True):

    param_cls = (
        (
            "from dataclasses import dataclass\n\n"
            "from bluemira.base.parameterframe import Parameter, ParameterFrame\n\n"
        )
        if header
        else ""
    )

    param_cls += "@dataclass\nclass {}(ParameterFrame):\n"

    if name is None:
        param_cls = param_cls.format("ReactorParams")
    else:
        param_cls = param_cls.format(name)

    param_row = "    {name}: {_type}\n"

    for param, param_type in params.items():
        param_cls += param_row.format(name=param, _type=str(param_type).split(".")[-1])

    return param_cls


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=str)
    parser.add_argument("-c", "--collapse", action="store_true")
    parser.add_argument("-d", "--directory", type=str, default="./")

    args = parser.parse_args()
    args.module = Path(args.module).resolve()
    return parser.parse_args()


def main():
    args = parse_args()
    module = get_module(args.module)

    param_classes = {
        f"{m[0]}: {m[1].param_cls.__name__}": m[1].param_cls
        for m in inspect.getmembers(module, inspect.isclass)
        if hasattr(m[1], "param_cls") and m[1].param_cls is not None
    }

    output = {}
    params = {}

    if args.collapse:
        for vv in param_classes.values():
            add_to_dict(vv, output, params)

        with open(Path(args.directory, "params.py"), "w") as fh:
            fh.write(create_parameterframe(params))
        json_writer(output, file=Path(args.directory, "params.json"))

    else:
        for kk, vv in param_classes.items():
            output[kk] = {}
            params[kk] = {}
            add_to_dict(vv, output[kk], params[kk])

        with open(Path(args.directory, "params.py"), "w") as fh:
            header = True
            for out_name, out_val in output.items():
                pname = out_name.replace(" ", "")
                pname = pname.replace(":", "_")
                fh.write(create_parameterframe(params[out_name], pname, header=header))
                fh.write("\n")
                json_writer(out_val, file=Path(args.directory, f"{pname}.json"))
                header = False


if __name__ == "__main__":
    main()
