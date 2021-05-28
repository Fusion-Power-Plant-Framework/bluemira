#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

# import freecad lib
import freecad
import Part

# import mesher lib (gmsh)
import gmsh

# import mirapy modules
from . import geo
from . import msh2xdmf
from . import core

import os


class Mesh():
    def __init__(
            self,
            modelname="Mesh",
            terminal=1,
            mesh_dim=3,
            meshfile="Mesh.geo_unrolled",
            embed=[]
    ):

        self.modelname = modelname
        self.terminal = terminal
        self.mesh_dim = mesh_dim
        self.meshfile = meshfile
        # embed is an array of tuple (point, lcar)
        self.embed = embed

    def _check_meshfile(self, meshfile):
        print(meshfile)
        if isinstance(meshfile, str):
            meshfile = [meshfile]
        elif isinstance(meshfile, list):
            if len(meshfile) < 1:
                raise ValueError("meshfile is an empty list")
        else:
            raise ValueError("meshfile must be a string or a list of strings")
        print("Meshfile = {}".format(meshfile))
        return meshfile

    @property
    def meshfile(self):
        return self._meshfile

    @meshfile.setter
    def meshfile(self, meshfile):
        self._meshfile = self._check_meshfile(meshfile)

    def __call__(self, obj, clean=True):
        if isinstance(obj, (geo.Shape, geo.Shape2D, core.Component)):
            self.__root = obj
            _freecadGmsh._initialize_mesh(self.terminal, self.modelname)
            self.__mesh_obj(obj, self.mesh_dim)
            self.__embed_points()
            self.__check_physical_groups(obj)
            _freecadGmsh._generate_mesh(self.mesh_dim)
            for file in self.meshfile:
                _freecadGmsh._save_mesh(file)
            _freecadGmsh._finalize_mesh()
            self.__clean_mesh_obj(obj, clean)
        else:
            raise ValueError("Only Shapes and Components can be meshed")

    def __mesh_obj(self, obj, mesh_dim=3):
        if isinstance(obj, geo.Shape):
            self.__mesh_shape(obj, mesh_dim)
        if isinstance(obj, geo.Shape2D):
            self.__mesh_shape2d(obj, mesh_dim)
            print(obj)
            print(obj.mesh_dict)
        if isinstance(obj, core.Component):
            self.__mesh_component(obj, mesh_dim)
            
        for dim in [0, 1]:
            print("dim = {}".format(dim))
            all_ent, oo, oov = _freecadGmsh._fragment([dim])

    def __clean_mesh_obj(self, obj, clean=True):
        if isinstance(obj, geo.Shape):
            self.__clean_mesh_shape(obj, clean)
        if isinstance(obj, geo.Shape2D):
            self.__clean_mesh_shape2d(obj, clean)
        if isinstance(obj, core.Component):
            self.__clean_mesh_component(obj, clean)

    def __mesh_shape(self, obj, mesh_dim=3):
        # print("Mesh shape1d {}".format(obj))
        if not hasattr(obj, "ismeshed") or not obj.ismeshed:
            obj.mesh_dict = []
            for o in obj.allshapes:
                if isinstance(o, Part.Wire):
                    obj.mesh_dict += _freecadGmsh._convert_wire_to_gmsh(
                        o, obj.lcar, mesh_dim)
                if isinstance(o, geo.Shape):
                    self.__mesh_shape(o, mesh_dim)
                    # fragment only made on points and curves
                    # if fragment in made also on surface, the construction
                    # points internal to a surface will be considered as
                    # "Point in Surface" (like Embedded points)
                    # try:
                    # all_ent,oo,oov=meshing._freecadGmsh._fragment([0,1,2])
                
                dim = max([k for k, t in self.get_mesh_dict(obj)])
                dimtags = [(dim, t) for t in self.get_mesh_dict(obj, dim)]
                all_ent = dimtags + gmsh.model.getBoundary(dimtags,
                                                           combined=False,
                                                           oriented=False,
                                                           recursive=True,
                                                           )
                # print(all_ent)
                all_ent, oo, oov = _freecadGmsh._fragment(all_ent=all_ent)
                obj.mesh_dict = _freecadGmsh._map_mesh_dict(obj.mesh_dict,
                                                            all_ent, oov)

            # # check fragment consistency only for the element of the shape
            # dim = max([k for k, t in self.get_mesh_dict(obj)])
            # print("dim: {}".format(dim))
            # print("mesh_dict: {}".format(self.get_mesh_dict(obj, dim)))
            # dimtags = [(dim, t) for t in self.get_mesh_dict(obj, dim)]
            # all_ent = dimtags + gmsh.model.getBoundary(dimtags,
            #                                            combined=True,
            #                                            oriented=False,
            #                                            recursive=True,
            #                                            )
            # print("all_ent: {}".format(all_ent))
            # all_ent, oo, oov = _freecadGmsh._fragment(all_ent=all_ent)
            # obj.mesh_dict = _freecadGmsh._map_mesh_dict(obj.mesh_dict,
            #                                             all_ent, oov)
            
            # all_ent = [(dim, t) for t in self.get_mesh_dict(obj, dim)]
            # print("all_ent shape: {}".format(all_ent))

        # for dim in [0, 1]:
        #     all_ent = [(dim, t) for t in self.get_mesh_dict(obj, dim)]
        #     if len(all_ent) > 1:
        #         all_ent, oo, oov = _freecadGmsh._fragment(all_ent=all_ent)
        #         obj.mesh_dict = _freecadGmsh._map_mesh_dict(obj.mesh_dict,
        #                                                     all_ent, oov)

            obj.ismeshed = True
            # print("obj.mesh_dict: {}".format(obj.mesh_dict))
        else:
            print("Obj already meshed")

    def __mesh_shape2d(self, obj, mesh_dim=3):
        # print("Mesh Shape2d {}".format(obj))
        
        # initialize ismeshed property
        if not hasattr(obj, "ismeshed"):
           obj.ismeshed = False 
        
        if not obj.ismeshed:
            obj.mesh_dict = []
            for o in obj.allshapes:
                self.__mesh_shape(o, mesh_dim)
                curvetag = [t for k, t in self.get_mesh_dict(o) if k == 1]
                looptag = gmsh.model.occ.addCurveLoop(curvetag)
                gmsh.model.occ.synchronize()
                obj.mesh_dict += [(-1, looptag)]
            # looptags = [(1,t) for k,t in obj.mesh_dict if k == -1]
            # surfs = [(2, gmsh.model.occ.addPlaneSurface([t]))
            #          for k,t in looptags]

            if mesh_dim > 1:
                looptags = [t for k, t in obj.mesh_dict if k == -1]
                surfs = (2, gmsh.model.occ.addPlaneSurface(looptags))
                gmsh.model.occ.synchronize()
                obj.mesh_dict += [surfs]

            # #The followings are two tests using gmsh.model.occ.remove
            # #Unfortunately it gives problems because the operations also
            # #changes the tag number of lines. However, it doesen't return
            # #the tagmap.

            # #Test1
            # if len(surfs)>1:
            #     surfs,tagmap = gmsh.model.occ.fragment(surfs[0:1], surfs[1:])
            #     gmsh.model.occ.synchronize()
            #     removetag =  list(itertools.chain.from_iterable(tagmap[1:]))
            #     removetag = list(dict.fromkeys(removetag))
            #     #remove the tag. Recursive is se to False
            #     gmsh.model.occ.remove(removetag, False)
            #     gmsh.model.occ.synchronize()
            #     [surfs.remove(t) for t in removetag]
            # obj.mesh_dict += surfs

            # #Test2
            # obj.mesh_dict += [surfs[0]]
            # if len(surfs)>1:
            #     all_ent, oo, oov = _freecadGmsh._fragment(all_ent=surfs[0:1],
            #                                               tools=surfs[1:])
            #     obj.mesh_dict = _freecadGmsh._map_mesh_dict(obj.mesh_dict,
            #                                                 all_ent, oov)
            #     print("surfs>1")
            #     print(all_ent)
            #     print(oov)

            #     removetag =  list(itertools.chain.from_iterable(oov[1:]))
            #     removetag = list(dict.fromkeys(removetag))
            #     #remove the tag. Recursive is se to False
            #     gmsh.model.occ.remove(removetag, True)
            #     gmsh.model.occ.synchronize()
            #     #[surfs.remove(t) for t in removetag]

            obj.ismeshed = True
            # print("obj.mesh_dict: {}".format(obj.mesh_dict))
        else:
            print("Obj already meshed")

    def __mesh_component(self, obj, mesh_dim=3):
        # only leaves are meshed
        leaves = obj.leaves
        for l in leaves:
            self.__mesh_obj(l.shape, mesh_dim)


        print("Meshing component {}".format(obj))
            
        # all_ent = [(k,v) for k,v in self.get_mesh_dict(obj) if k>=0]
        # all_ent, oo, oov = _freecadGmsh._fragment([0,1,2], all_ent)

        # for l in leaves:
        #     try:
        #         l.shape.mesh_dict = _freecadGmsh._map_mesh_dict(
        #             l.shape.mesh_dict, all_ent, oov)
        #     except:
        #         pass

        # # to check duplication of construction points
        # all_ent, oo, oov = _freecadGmsh._fragment([0])
        # for l in leaves:
        #     try:
        #         # print(l.shape.mesh_dict)
        #         l.shape.mesh_dict = _freecadGmsh._map_mesh_dict(
        #             l.shape.mesh_dict, all_ent, oov)
        #         # print(l.shape.mesh_dict)
        #     except:
        #         pass

    def __clean_mesh_shape(self, obj, clean=True):
        if hasattr(obj, 'ismeshed'):
            delattr(obj, 'ismeshed')
            if clean:
                delattr(obj, 'mesh_dict')

            if hasattr(obj, 'addedphysicalGroups'):
                delattr(obj, 'addedphysicalGroups')

            for o in obj.allshapes:
                if isinstance(o, geo.Shape):
                    self.__clean_mesh_shape(o, clean)

    def __clean_mesh_shape2d(self, obj, clean=True):
        if hasattr(obj, 'ismeshed'):
            delattr(obj, 'ismeshed')
            if clean:
                delattr(obj, 'mesh_dict')

            if hasattr(obj, 'addedphysicalGroups'):
                delattr(obj, 'addedphysicalGroups')

            for o in obj.allshapes:
                if isinstance(o, geo.Shape):
                    self.__clean_mesh_shape(o, clean)

    def __clean_mesh_component(self, obj, clean=True):
        leaves = obj.leaves
        for l in leaves:
            self.__clean_mesh_shape(l.shape, clean)

    def get_mesh_dict(self, obj, dim=None):
        mesh_dict = []
        if hasattr(obj, "mesh_dict"):
            mesh_dict += obj.mesh_dict
        if isinstance(obj, geo.Shape):
            for o in obj.boundary:
                mesh_dict += self.get_mesh_dict(o)
        if isinstance(obj, geo.Shape2D):
            for o in obj.allshapes:
                mesh_dict += self.get_mesh_dict(o)
        if isinstance(obj, core.Component):
            if obj.shape:
                mesh_dict += self.get_mesh_dict(obj.shape)
            if not obj.is_leaf:
                for l in obj.leaves:
                    mesh_dict += self.get_mesh_dict(l)
        if dim is not None:
            mesh_dict = [v for k, v in mesh_dict if k == dim]
        return mesh_dict

    def __check_physical_groups(self, obj):
        if hasattr(obj, "physicalGroups"):
            if (not hasattr(obj, "addedphysicalGroups") or
                    not obj.addedphysicalGroups):
                obj.addedphysicalGroups = []

            for dim, name in obj.physicalGroups.items():
                if dim not in obj.addedphysicalGroups:
                    print("add_physical_group {}".format(name))
                    mesh_dict = self.get_mesh_dict(obj)
                    _freecadGmsh._add_physical_group(mesh_dict, dim, name)
                    obj.addedphysicalGroups += [dim]

        if isinstance(obj, geo.Shape):
            for i in obj.allshapes:
                self.__check_physical_groups(i)
        if isinstance(obj, geo.Shape2D):
            for i in obj.allshapes:
                self.__check_physical_groups(i)
        if isinstance(obj, core.Component):
            if obj.shape:
                self.__check_physical_groups(obj.shape)
            if not obj.is_leaf:
                # print("This condition should not happen.
                # Only leaves are meshed!")
                [self.__check_physical_groups(l) for l in obj.leaves]

    def __embed_points(self):
        if self.embed:
            for p, l in self.embed:
                _freecadGmsh._embed_point(p, l)


class _freecadGmsh():

    @staticmethod
    def _initialize_mesh(terminal=1, modelname="Mesh"):

        # GMSH file generation #######################
        # Before using any functions in the Python API,
        # Gmsh must be initialized:
        gmsh.initialize()

        # By default Gmsh will not print out any messages:
        # in order to output messages
        # on the terminal, just set the "General.Terminal" option to 1:
        gmsh.option.setNumber("General.Terminal", terminal)

        # Next we add a new model named "t1" (if gmsh.model.add() is
        # not called a new
        # unnamed model will be created on the fly, if necessary):
        gmsh.model.add(modelname)

    @staticmethod
    def _save_mesh(meshfile="Mesh.geo_unrolled"):
        # ... and save it to disk
        gmsh.write(meshfile)

    @staticmethod
    def _finalize_mesh():
        # This should be called when you are done using the Gmsh Python API:
        gmsh.finalize()

    @staticmethod
    def _generate_mesh(mesh_dim=3):
        # Before it can be meshed, the internal CAD representation must
        # be synchronized with the Gmsh model, which will create the
        # relevant Gmsh data structures. This is achieved by the
        # gmsh.model.occ.synchronize() API call for the built-in
        # geometry kernel. Synchronizations can be called at any time,
        # but they involve a non trivial amount of processing;
        # so while you could synchronize the internal CAD data after
        # every CAD command, it is usually better to minimize
        # the number of synchronization points.
        gmsh.model.occ.synchronize()

        # We can then generate a mesh...
        gmsh.model.mesh.generate(mesh_dim)

    @staticmethod
    def _fragment(dim=[0, 1, 2], all_ent=None, tools=[]):
        if not hasattr(dim, '__len__'):
            dim = [dim]
        if all_ent is None:
            all_ent = []
            for d in dim:
                all_ent += gmsh.model.getEntities(d)

        # print("all_ent:{}".format(all_ent))
        # print("tools:{}".format(tools))
        
        oo = []
        oov = []
        if len(all_ent) > 1:
            oo, oov = gmsh.model.occ.fragment(all_ent, tools)
            gmsh.model.occ.synchronize()
    
            # print("oo:{}".format(oo))
            # print("oov:{}".format(oov))

        return all_ent, oo, oov

    @staticmethod
    def _map_mesh_dict(mesh_dict, all_ent, oov):
        tags = []
        for k, v in mesh_dict:
            if (k, v) in all_ent:
                if len(oov) > 0:
                    for o in oov[all_ent.index((k, v))]:
                        tags.append(o)
            else:
                tags.append((k, v))
        return tags

    @staticmethod
    def _convert_wire_to_gmsh(wire, lcar=0.1, mesh_dim=1):

        points = []
        ends = []
        pointstag = []
        constructionpoints = []
        constructionpointstag = []
        curves = []
        curvestag = []

        isclosed = wire.isClosed()

        edges = wire.OrderedEdges
        l = len(edges)

        if not hasattr(lcar, "__len__"):
            lcar = [lcar]*l
        else:
            if l > len(lcar):
                lcar = lcar + [lcar[-1]]*(l - len(lcar))

        if(l != len(lcar)):
            raise ValueError("l not equal to len(lcar)"
                             " == {}".format(len(lcar)))

        curves = geo.Utils.convert_Wire_to_Curves(wire)
        # [print("{}: {}".format(e, e.Orientation)) for e in edges]
        # [print("{}: {} {}".format(c, c.StartPoint, c.EndPoint))
         # for c in curves]
        if mesh_dim >= 0:
            for i in range(l):

                p = curves[i].StartPoint
                points.append(p)

                ends.append(curves[i].EndPoint)

                if i == (l-1):
                    if not isclosed:
                        points.append(curves[i].EndPoint)

            for i in range(len(points)):
                p = points[i]
                if i < len(lcar):
                    pointstag.append(gmsh.model.occ.addPoint(p.x,
                                                             p.y,
                                                             p.z,
                                                             lcar[i],
                                                             )
                                     )
                else:
                    pointstag.append(gmsh.model.occ.addPoint(p.x,
                                                             p.y,
                                                             p.z,
                                                             lcar[-1],
                                                             )
                                     )
                    lcar.append(lcar[-1])

            if mesh_dim >= 1:
                for i in range(l):
                    if isclosed:
                        j = (i+1) % l
                    else:
                        j = i+1
                    curve = curves[i]

                    if isinstance(curve, (Part.LineSegment, Part.Line)):
                        curvestag.append(gmsh.model.occ.addLine(pointstag[i],
                                                                pointstag[j],
                                                                )
                                         )

                    elif isinstance(curve, Part.BSplineCurve):
                        poles = curve.getPoles()
                        poles = poles[1:-1]
                        controlpoints = [points[i]]
                        controlpointstag = [pointstag[i]]

                        for p in poles:
                            controlpoints.append(p)
                            controlpointstag.append(
                                gmsh.model.occ.addPoint(p.x, p.y, p.z))

                        controlpoints.append(points[j])
                        controlpointstag.append(pointstag[j])

                        curvestag.append(gmsh.model.occ.addBSpline(
                            [cptag for cptag in controlpointstag]))

                        constructionpoints += controlpoints
                        constructionpointstag += controlpointstag

                    elif isinstance(curve, Part.BezierCurve):
                        poles = curve.getPoles()
                        poles = poles[1:-1]
                        controlpoints = [points[i]]
                        controlpointstag = [pointstag[i]]

                        for p in poles:
                            controlpoints.append(p)
                            controlpointstag.append(
                                gmsh.model.occ.addPoint(p.x, p.y, p.z))

                        controlpoints.append(points[j])
                        controlpointstag.append(pointstag[j])

                        curvestag.append(gmsh.model.occ.addBezier(
                            [cptag for cptag in controlpointstag]))

                        constructionpoints += controlpoints
                        constructionpointstag += controlpointstag

                    elif isinstance(curve, (Part.ArcOfCircle)):
                        p = curve.Center
                        centertag = gmsh.model.occ.addPoint(p.x, p.y, p.z)
                        curvestag.append(gmsh.model.occ.addCircleArc(
                            pointstag[i],
                            centertag,
                            pointstag[j]))
                        constructionpoints.append(p)
                        constructionpointstag.append(centertag)

                    elif isinstance(curve, (Part.ArcOfEllipse)):
                        p = curve.Center
                        centertag = gmsh.model.occ.addPoint(p.x, p.y, p.z)
                        f1 = curve.Ellipse.Focus1
                        focus1tag = gmsh.model.occ.addPoint(f1.x, f1.y, f1.z)

                        curvestag.append(gmsh.model.occ.addEllipseArc(
                            pointstag[i],
                            centertag,
                            focus1tag,
                            pointstag[j]))

                        constructionpoints.append(p)
                        constructionpointstag.append(centertag)
                        constructionpoints.append(f1)
                        constructionpointstag.append(focus1tag)

                    else:
                        print("{} is still not supported!".format(type(curve)))

        gmsh.model.occ.synchronize()
        # construction points (as for Circles and Ellipses) are not stored
        # impot the object gmsh_dict
        gmsh_dict = [(0, p) for p in pointstag] + [(1, c) for c in curvestag]
        return gmsh_dict

    @staticmethod
    def _add_physical_group(mesh_dict, dim, name):
        try:
            tags = [v for k, v in mesh_dict if k == dim]
            phys_tag = gmsh.model.addPhysicalGroup(dim, tags)
            gmsh.model.setPhysicalName(dim, phys_tag, name)
            gmsh.model.occ.synchronize()
            # print("Obj {} mesh_dict: {}".format(obj, obj.mesh_dict))
        except:
            print("Warning: Obj not meshed or wrong obj.mesh_dict")

    @staticmethod
    def _embed_point(p, lcar=0.1):
        # Some problems with this method. I would like to embed the point
        # only if the point is inside the surface, but both the
        # gmsh.model.isInside and gmsh.model.getClosestPoint doesn't
        # give the expected result. Probably I am missing something.

        pointtag = gmsh.model.occ.addPoint(p[0], p[1], p[2], lcar)
        gmsh.model.occ.synchronize()

        surf = gmsh.model.getEntities(2)
        print("Embed point {} in surf {}".format(p, surf))
        for s in surf:
            gmsh.model.mesh.embed(0, [pointtag], 2, s[1])

        gmsh.model.occ.synchronize()


def get_mesh_dict(obj, dim=None):
    mesh_dict = []
    if hasattr(obj, "mesh_dict"):
        mesh_dict += obj.mesh_dict
    if isinstance(obj, geo.Shape):
        for o in obj.objs:
            mesh_dict += get_mesh_dict(o)
    if isinstance(obj, geo.Shape2D):
        mesh_dict = obj.mesh_dict
        for o in obj.objs + obj.holes:
            mesh_dict += get_mesh_dict(o)
    if isinstance(obj, geo.Component):
        if obj.shape:
            mesh_dict += get_mesh_dict(obj.shape)
        if not obj.is_leaf:
            for l in obj.leaves:
                mesh_dict += get_mesh_dict(l)
    if dim is not None:
        mesh_dict = [v for k, v in mesh_dict if k == dim]
    return mesh_dict

def map_mesh_dict(obj, all_ent, oov):
    if hasattr(obj, "mesh_dict"):
        obj.mesh_dict = _freecadGmsh.map_mesh_dict(obj.mesh_dict,
                                                   all_ent,
                                                   oov
                                                   )
    if isinstance(obj, geo.Shape):
        for o in obj.allshapes:
            map_mesh_dict(o, all_ent, oov)
    if isinstance(obj, geo.Shape2D):
        for o in obj.allshapes:
            map_mesh_dict(o, all_ent, oov)
    if isinstance(obj, core.Component):
        if obj.shape:
            map_mesh_dict(obj.shape, all_ent, oov)
        if not obj.is_leaf:
            for l in obj.leaves:
                map_mesh_dict(l, all_ent, oov)
    return None

def setPhysicalGroups(component):
    if component.shape is not None:
        str1 = component.name + "1D"
        str2 = component.name + "2D"
        if (not hasattr(component.shape, 'physicalGroups') or
             component.shape.physicalGroups is None):
            component.shape.physicalGroups = {1: str1, 2: str2}
        else:
            if 1 not in component.shape.physicalGroups:
                component.shape.physicalGroups[1] = str1
            if 2 not in component.shape.physicalGroups:
                component.shape.physicalGroups[2] = str2
    if not component.is_leaf:
        for l in component.leaves:
            setPhysicalGroups(l)


def getDolfinMesh(component, meshfile="Mesh.msh", meshdir=".", mesh_dim=2):
    setPhysicalGroups(component)

    mesh = Mesh("Component")
    mesh.meshfile = os.path.join(meshdir, meshfile)
    mesh(component)

    # Run the conversion
    msh2xdmf.msh2xdmf(meshfile, dim=mesh_dim, directory=meshdir)

    # Run the import
    prefix, _ = os.path.splitext(meshfile)

    # Read mesh from xdmf
    mesh, boundaries, subdomains, labels = \
        msh2xdmf.import_mesh_from_xdmf(
            prefix=prefix,
            dim=mesh_dim,
            directory=meshdir,
            subdomains=True,
        )

    return mesh, boundaries, subdomains, labels
