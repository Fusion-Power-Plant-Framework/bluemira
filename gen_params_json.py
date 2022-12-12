import inspect
from pathlib import Path

from bluemira.base.parameterframe._parameter import ParamDictT
from bluemira.utilities.tools import get_module, json_writer

DEFAULT_PARAM = ParamDictT.__annotations__


def add_to_dict(vv, out_dict, params):
    for param, param_type in vv.__annotations__.items():
        dv = deepcopy(DEFAULT_PARAM)
        dv["value"] = param_type
        out_dict[param] = dv
        params[param] = param_type


def create_parameterframe(params, name=None):

    param_cls = (
        "from dataclasses import dataclass\n\n"
        "from bluemira.base.parameterframe import Parameter, ParameterFrame\n\n"
        "@dataclass\nclass {}Params(ParameterFrame):\n"
    )

    if name is None:
        param_cls = param_cls.format("Reactor")
    else:
        param_cls = param_cls.format(name)

    param_row = "    {name}: Parameter[{_type}]\n"

    for param, param_type in params.items():
        param_cls += param_row.format(name=param, _type=param_type)

    return param_cls


def parse_args():
    raise NotImplementedError


def main():
    args = parse_args()
    module = get_module(args.module)
    param_classes = {
        f"{m[0].__name__}: {m[0].param_cls.__name__}": m[0].param_cls
        for m in inspect.getmembers(module, inspect.isclass)
        if hasattr(m[0], "param_cls")
    }

    output = {}
    params = {}

    if args.collapse:
        for vv in param_classes.values():
            add_to_dict(vv, output, params)

        with open(Path(args.location, "params.py"), w) as fh:
            fh.write(create_parameterframe(params))

        json_writer(output, file=Path(args.location, "params.json"))

    else:
        for kk, vv in param_classes.items():
            output[kk] = {}
            params[kk] = {}
            add_to_dict(vv, output[kk], params[kk])

        with open(Path(args.location, "params.py"), w) as fh:
            for out_name, out_val in output.items():
                pname = out_name.replace(" ", "")
                pname = pname.replace(":", "_")
                fh.write(create_parameterframe(params[out_name], pname))
                fh.write("\n")
                json_writer(out_val, file=Path(args.location, f"{pname}.json"))
