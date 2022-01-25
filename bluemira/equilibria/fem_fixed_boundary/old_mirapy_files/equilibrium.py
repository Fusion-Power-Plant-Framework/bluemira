#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 23:27:04 2020

@author: ivan
"""
import os

import freecad
import anytree
import dolfin
import matplotlib.pyplot as plt
import numpy
import Part
import scipy
import scipy.interpolate
from FreeCAD import Base
from scipy.integrate import quad

import mirapy


class equilibrium(object):
    """
    Equilibrium class
    """

    def __init__(self, nsect):
        self.__nsect = nsect
        self.__default_filter = lambda node: (
            hasattr(node, "filaments") and not node.filaments is None
        )
        self.set_to_default()

    def __call__(self, tokamak):
        self.__tokamak = tokamak

    @property
    def tokamak(self):
        return self.__tokamak

    @property
    def nsect(self):
        return self.__nsect

    def select_nodes(self, nodefilter):
        nodes = anytree.search.findall(self.__tokamak, filter_=nodefilter)
        return nodes

    def selected_nodes(self):
        return self.select_nodes(self.__filter)

    @property
    def nodefilter(self):
        return self.__filter

    @nodefilter.setter
    def nodefilter(self, value):
        self.__filter = value

    def set_to_default(self):
        self.__filter = self.__default_filter
        self.__solvers = {}

    def calculateB(self, points3D, total=True, green=False):
        nodes = self.select_nodes(self.__filter)
        B = numpy.asarray([n.filaments.calculateB(points3D, green) for n in nodes])
        if total:
            B = sum(B)
        return B

    def calculatePsi(self, points3D, total=True, green=False):
        nodes = self.select_nodes(self.__filter)
        Psi = numpy.asarray([n.filaments.calculatePsi(points3D, green) for n in nodes])
        if total:
            Psi = sum(Psi)
        return Psi

    def ripple(self, pointsXY, ndiscr=3):
        filter_ = lambda node: isinstance(node, mirapy.core.TFCoil)
        nodes = self.select_nodes(filter_)
        ripple = numpy.zeros(pointsXY.shape[0])
        points = [Part.Point(Base.Vector(p)) for p in pointsXY]
        for i in range(len(points)):
            p = points[i]
            B = numpy.zeros((ndiscr, 3))

            toroidalPoints = []
            angles = numpy.linspace(0.0, 360.0 / self.__nsect, num=ndiscr)

            for alpha in angles:
                newp = p.copy()
                bp = Base.Placement(Base.Vector(), Base.Vector(0, 1, 0), alpha)
                newp.rotate(bp)
                toroidalPoints.append(newp)

            points3D = numpy.array([[v.X, v.Y, v.Z] for v in toroidalPoints])
            for n in nodes:
                B += n.filaments.calculateB(points3D)

            rotateB = [Part.Point(Base.Vector(v)) for v in B]

            for j in range(len(rotateB)):
                rotateB[j].rotate(
                    Base.Placement(Base.Vector(), Base.Vector(0, 1, 0), -angles[j])
                )

            rotateB = numpy.array([[v.X, v.Y, v.Z] for v in rotateB])

            Bz = rotateB[:, 2]

            Bmax = max(Bz)
            Bmin = min(Bz)

            ripple[i] = abs((Bmax - Bmin) / (Bmax + Bmin))
        return ripple

    def __getClass(self, cls_name):
        filter_ = lambda node: isinstance(node, cls_name)
        nodes = self.select_nodes(filter_)
        return nodes

    def getPlasma(self):
        plasma = self.__getClass(mirapy.core.Plasma)
        if len(plasma) != 1:
            print("Warning: found zero or more than 1 plasma. None is returned.")
            return None
        return plasma[0]

    def getTFCoils(self):
        return self.__getClass(mirapy.core.TFCoil)

    def getPFCoils(self):
        return self.__getClass(mirapy.core.PFCoil)

    def getCoils(self):
        return self.__getClass(mirapy.core.Coil)

    def plasma_fixed_boundary(
        self,
        createmesh=True,
        meshfile="Mesh.msh",
        meshdir=".",
        Pax=None,
        Pax_lcar=0.1,
        maxiter=100,
        tol=1e-6,
        p=5,
        solver=None,
    ):

        plasma = self.getPlasma()

        if plasma is None:
            raise ValueError("No plasma has been found")

        if solver is None:
            if (
                not hasattr(plasma.shape, "physicalGroups")
            ) or plasma.shape.physicalGroups is None:
                plasma.shape.physicalGroups = {1: "external", 2: "plasma"}
            else:
                if not 1 in plasma.shape.physicalGroups:
                    plasma.shape.physicalGroups[1] = "external"

                if not 2 in plasma.shape.physicalGroups:
                    plasma.shape.physicalGroups[2] = "plasma"

            mesh_dim = 2

            if plasma.J is None:
                raise ValueError("Plamsa Jp must to be defined")

            fullmeshfile = os.path.join(meshdir, meshfile)

            print(fullmeshfile)

            if createmesh:
                #### Mesh Generation ####
                mesh = mirapy.core.Mesh("plasma")
                mesh.meshfile = fullmeshfile
                if not Pax is None:
                    # P0lcar = plasma.shape.lcar/2.
                    mesh.embed = [(Pax, Pax_lcar)]
                mesh(plasma)

            # Run the conversion
            mirapy.msh2xdmf.msh2xdmf(meshfile, dim=mesh_dim)

            # Run the import
            prefix, _ = os.path.splitext(fullmeshfile)

            mesh, boundaries, subdomains, labels = mirapy.msh2xdmf.import_mesh_from_xdmf(
                prefix=prefix,
                dim=mesh_dim,
                directory=meshdir,
                subdomains=True,
            )

            solver = mirapy.dolfinSolver.GradShafranovLagrange(mesh, p=p)

        # Calculate plasma geometrical parameters
        plasma.calculatePlasmaParameters(solver.mesh)

        eps = 1.0  # error measure ||u-u_k||
        i = 0  # iteration counter
        while eps > tol and i < maxiter:
            prev = solver.psi.compute_vertex_values()
            i += 1
            plasma.psi = solver.psi
            g = plasma.J_to_dolfinFunction(solver.V)
            solver.solve(g)
            diff = solver.psi.compute_vertex_values() - prev
            eps = numpy.linalg.norm(diff, ord=numpy.Inf)
            print("iter = {} eps = {}".format(i, eps))
            plasma.dolfinUpdate(solver.V)

        self.__solvers["fixed_boundary"] = solver
        plasma.updateFilaments(solver.V)

    def get_solver(self, label=None):
        if label is None:
            return self.__solvers
        return self.__solvers[label]
