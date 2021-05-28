#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for mirapy.emag

@author: ivan
"""
import pytest

import mirapy.emag as emag
import numpy as np


class TestEmagMethods:
    def test_Utils_Bteo_coil(self):
        """
        Test B calculation of a coil along the axis
        """
        rc = 1.0
        zc = 0.0
        pr = 0.0
        pz = 0.0
        Ic = 1.0e6

        B = emag.Utils.Bteo_coil(rc, zc, pr, pz, Ic)
        B == pytest.approx(0.6283185307179)

    def test_Green_calculatePsi(self):
        """
        Test Green.calculatePsi method
        """
        Rc = np.array([1, 2])
        Zc = np.array([0, 0])
        R = np.array([0, 1, 2])
        Z = np.array([0, 1, 0])

        psi = emag.Greens.calculatePsi(Rc, Zc, R, Z)
        assert len(psi) == 3


if __name__ == "__main__":
    pytest.main([__file__])
