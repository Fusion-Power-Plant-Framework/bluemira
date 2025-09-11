# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


import numpy as np
import pytest
from eurofusion_materials.library.magnet_branch_mats import (
    NB3SN_MAG,
    SS316_LN_MAG,
)
from matproplib import OperationalConditions
from matproplib.material import MaterialFraction

from bluemira.magnets.strand import Strand, SuperconductingStrand

# %%

# get some materials from MATERIAL_CACHE
DummySteel = SS316_LN_MAG
DummySuperconductor1 = NB3SN_MAG
DummySuperconductor2 = NB3SN_MAG


def test_strand_area():
    mat = MaterialFraction(material=DummySuperconductor1, fraction=1.0)
    strand = Strand(name="test_strand", materials=[mat], d_strand=0.001)
    expected_area = np.pi * (0.001**2) / 4
    assert np.isclose(strand.area, expected_area)


def test_strand_invalid_diameter():
    mat = MaterialFraction(material=DummySuperconductor1, fraction=1.0)
    with pytest.raises(ValueError, match="positive"):
        Strand(name="invalid_strand", materials=[mat], d_strand=-0.001)


def test_superconducting_strand_invalid_materials():
    # Two superconductors — should raise ValueError
    mat1 = MaterialFraction(material=DummySuperconductor1, fraction=0.5)
    mat2 = MaterialFraction(material=DummySuperconductor2, fraction=0.5)
    with pytest.raises(ValueError, match="Only one superconductor material"):
        SuperconductingStrand(name="invalid", materials=[mat1, mat2])

    # No superconductors — should raise ValueError
    mat3 = MaterialFraction(material=DummySteel, fraction=1.0)
    with pytest.raises(ValueError, match="No superconducting material"):
        SuperconductingStrand(name="invalid", materials=[mat3])


def test_strand_material_properties():
    sc = DummySuperconductor1
    mat = MaterialFraction(material=sc, fraction=1.0)
    strand = Strand(name="mat_test", materials=[mat], d_strand=0.001)

    temperature = 20
    op_cond = OperationalConditions(temperature=20)
    assert np.isclose(strand.erho(op_cond), sc.electrical_resistivity(op_cond))

    assert np.isclose(strand.Cp(op_cond), sc.specific_heat_capacity(op_cond))
    assert np.isclose(strand.rho(op_cond), sc.density(op_cond))
