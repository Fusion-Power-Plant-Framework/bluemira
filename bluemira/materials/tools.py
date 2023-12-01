import json
import warnings


def import_nmm():
    """Don't hack my json, among other annoyances."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        import neutronics_material_maker as nmm

        # Really....
        json.JSONEncoder.default = nmm.material._default.default

    return nmm
