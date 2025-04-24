# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import numpy as np
import pytest

from bluemira.magnets.strand import Strand, SuperconductingStrand
from bluemira.materials import MaterialCache
from bluemira.materials.mixtures import MixtureFraction

# %%

# load supporting bluemira materials
MATERIAL_CACHE = MaterialCache()
MATERIAL_CACHE.load_from_file(Path(".", "test_materials_mag.json"))

# get some materials from MATERIAL_CACHE
DummySteel = MATERIAL_CACHE.get_material("SS316-LN")
DummySuperconductor1 = MATERIAL_CACHE.get_material("Nb3Sn - WST")
DummySuperconductor2 = MATERIAL_CACHE.get_material("Nb3Sn - WST")


def test_strand_area():
    mat = MixtureFraction(material=DummySuperconductor1, fraction=1.0)
    strand = Strand("test_strand", materials=[mat], d_strand=0.001)
    expected_area = np.pi * (0.001**2) / 4
    assert np.isclose(strand.area, expected_area)


def test_strand_invalid_diameter():
    mat = MixtureFraction(material=DummySuperconductor1, fraction=1.0)
    with pytest.raises(ValueError, match="positive"):
        Strand("invalid_strand", materials=[mat], d_strand=-0.001)


def test_superconducting_strand_invalid_materials():
    # Two superconductors — should raise ValueError
    mat1 = MixtureFraction(material=DummySuperconductor1, fraction=0.5)
    mat2 = MixtureFraction(material=DummySuperconductor2, fraction=0.5)
    with pytest.raises(ValueError, match="Only one superconductor material"):
        SuperconductingStrand("invalid", materials=[mat1, mat2])

    # No superconductors — should raise ValueError
    mat3 = MixtureFraction(material=DummySteel, fraction=1.0)
    with pytest.raises(ValueError, match="No superconducting material"):
        SuperconductingStrand("invalid", materials=[mat3])


def test_strand_material_properties():
    sc = DummySuperconductor1
    mat = MixtureFraction(material=sc, fraction=1.0)
    strand = Strand("mat_test", materials=[mat], d_strand=0.001)

    temperature = 20
    assert np.isclose(
        strand.erho(temperature=temperature), sc.erho(temperature=temperature)
    )
    print(sc.Cp(temperature=temperature))
    print(strand.Cp(temperature=temperature))

    assert np.isclose(strand.Cp(temperature=temperature), sc.Cp(temperature=temperature))
    assert np.isclose(
        strand.rho(temperature=temperature), sc.rho(temperature=temperature)
    )
