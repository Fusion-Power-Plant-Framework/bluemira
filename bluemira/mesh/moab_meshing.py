from collections.abc import Iterable
from pathlib import Path

from bluemira.geometry.base import BluemiraGeoT
from bluemira.geometry.overlap_checking import find_approx_overlapping_pairs
from bluemira.geometry.solid import BluemiraSolid


def imprint_solids(solids: Iterable[BluemiraSolid]):
    """Imprints solids together."""
    pairs = find_approx_overlapping_pairs(solids)


def save_cad_to_dagmc_model(
    shapes: Iterable[BluemiraGeoT],
    names: list[str],
    filename: Path,
    *,
    faceting_tolerance=0.001,
):
    """Converts the shapes with their associated names to a dagmc file using PyMOAB."""
