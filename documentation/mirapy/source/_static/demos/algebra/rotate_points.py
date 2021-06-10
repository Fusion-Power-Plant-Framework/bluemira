#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""
import mirapy.algebra as algebra

angle = 90.
raxis = 'x'
order = 0

points3D = [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)]

newp = algebra.rotate_points(points3D, angle, raxis, order)