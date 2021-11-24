from bluemira.base.config import Configuration

params = {}
for param in Configuration.params:
    if param[4] is None and len(param) == 6:
        params[param[0]] = param[2]
    else:
        params[param[0]] = {}
        params[param[0]]["value"] = param[2]
        if param[4] is not None:
            params[param[0]]["description"] = param[4]
        if len(param) == 7:
            params[param[0]]["mapping"] = {
                key: value.to_dict() for key, value in param[6].items()
            }

build_config = {
    "process_mode": "run",
}
