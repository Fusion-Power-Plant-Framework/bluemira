Creation
--------
A basic example for the creation of the geometrical objects:

* a `BluemiraWire` through a list of points:

    .. code-block:: pycon

        >>> from bluemira.geometry.tools import make_polygon
        >>> pntslist = [(1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0,.0)]
        >>> wire1 = make_polygon(pntslist, label="wire1")
        >>> print(wire1)
        ([BluemiraWire] = Label: wire1, length: 3.0, area: 0.0, volume: 0.0)


    It is possible to force the closure of the wire setting the parameter `closed` of
    ``make_polygon`` to ``True``.

    .. code-block:: pycon

        >>> pntslist = [(1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        >>> wire1 = make_polygon(pntslist, label="wire1", closed=True)
        >>> print(wire1)
        ([BluemiraWire] = Label: wire1, length: 4.0, area: 0.0, volume: 0.0)
        >>> wire1.is_closed()
        True

    The same procedure can be applied to ``make_bezier`` and ``make_bspline`` to generate
    bezier and bspline curves, respectively.

* a `BluemiraFace`:

    A BluemiraFace object is defined by its boundary that must be a closed
    BluemiraWire. If a list of BlumiraWire (with more than one wire) is given, the
    first wire is used as outer boundary of the face, while the others are considered
    as face holes.

    .. warning:: wires representing holes must not intersect. No internal check is
        implemented for the moment, so the check is on the user.

    .. code-block:: pycon

        >>> from bluemira.geometry.face import BluemiraFace
        >>> from bluemira.geometry.tools import make_polygon

        >>> pntslist_out = [(1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0,0.0)]
        >>> delta = 0.25
        >>> pntslist_in = [
        ...                 (1.0 - delta, 1.0 - delta, 0.0),
        ...                 (0.0 + delta, 1.0 - delta, 0.0),
        ...                 (0.0 + delta, 0.0 + delta, 0.0),
        ...                 (1.0 - delta, 0.0 + delta, 0.0),
        ...               ]
        >>> wire_out = make_polygon(pntslist_out, label="wire_out",closed=True)
        >>> bmface = BluemiraFace(wire_out)
        >>> print(bmface)
        ([BluemiraFace] = Label: , length: 4.0, area: 1.0, volume: 0.0)
        >>> wire_in = make_polygon(pntslist_in, label="wire_in", closed=True)
        >>> bmface_with_hole = BluemiraFace([wire_out, wire_in],label="face_with_hole")
        >>> print(bmface_with_hole)
        ([BluemiraFace] = Label: face_with_hole, length: 6.0, area: 0.75, volume: 0.0)

    .. note:: the length of the face is equal to the total length of the boundary.

* a `BluemiraShell`

    A BluemiraShell object is defined by its boundary that must be a set of
    BluemiraFace objects.

    .. warning:: faces shall not intersect. No internal check is implemented for the
        moment, so the check is on the user.

    .. code-block:: python

        from operator import itemgetter
        from bluemira.geometry.face import BluemiraFace
        from bluemira.geometry.shell import BluemiraShell

        vertexes = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)]
        # faces creation
        faces = []
        v_index = [(0,1,2,3),(5,4,7,6),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]
        for ind, value in enumerate(v_index):
            wire = make_polygon(list(itemgetter(*value)(vertexes)),closed=True)
            faces.append(BluemiraFace(wire, "face"+str(ind)))
        # shell creation
        shell = BluemiraShell(faces, "shell")


* a `BluemiraSolid`

    A BluemiraSolid object is defined by its boundary that must be a closed
    BluemiraShell object.

    .. code-block:: python

        from bluemira.geometry.solid import BluemiraSolid

        # solid creation from shell
        solid = BluemiraSolid(shell, "solid")

Shape operations
----------------
Shape operations that modify the shape itself are implemented as methods. For
example, the following command applies a translation with the specified vector:

    .. code-block:: python

        vector = (5.0, 2.0, 0.0)
        bmface.translate(vector)

Shape operations that, when applied, create a new shape topology are implemented in
``bluemira.geometry.tools``. For example, the following command creates a solid
by revolving a face of 30 degrees along the z-axis:

    .. code-block:: python

        from bluemira.geometry.tools import revolve_shape
        base = (0., 0., 0.)
        direction = (0., 0., 1.)
        degree = 30
        bmsolid = revolve_shape(bmface, base, direction, degree)

Exporting
---------
Each bluemira geometry object can be exported as step file:

    .. code-block:: python

        from bluemira.geometry.tools import save_cad

        save_cad(bmface, (tmp_path/"face.step").as_posix())


FreeCAD objects
---------------

Below a list of typical geometry object properties of FreeCAD. Only those labelled as
`converted` are available in bluemira.

    .. code-block:: python

        ['Area',  # converted
         'BoundBox',  # converted
         'CenterOfMass',  # converted
         'CompSolids',
         'Compounds',
         'Content',
         'Continuity',
         'Edges',
         'Faces',
         'Length',  # converted
         'Mass',
         'Matrix',
         'MatrixOfInertia',
         'MemSize',
         'Module',
         'OrderedEdges',
         'OrderedVertexes',
         'Orientation',
         'Placement',
         'PrincipalProperties',
         'ShapeType',
         'Shells',
         'Solids',
         'StaticMoments',
         'SubShapes',
         'Tag',
         'TypeId',
         'Vertexes',
         'Volume',  # converted
         'Wires',
         '__class__',
         '__delattr__',
         '__dir__',
         '__doc__',
         '__eq__',
         '__format__',
         '__ge__',
         '__getattribute__',
         '__getstate__',
         '__gt__',
         '__hash__',
         '__init__',
         '__init_subclass__',
         '__le__',
         '__lt__',
         '__ne__',
         '__new__',
         '__reduce__',
         '__reduce_ex__',
         '__repr__',
         '__setattr__',
         '__setstate__',
         '__sizeof__',
         '__str__',
         '__subclasshook__',
         'add',
         'ancestorsOfType',
         'approximate',
         'check',
         'childShapes',
         'cleaned',
         'common',
         'complement',
         'copy',
         'countElement',
         'cut',
         'defeaturing',
         'discretize',  # converted/improved
         'distToShape',
         'dumpContent',
         'dumpToString',
         'exportBinary',
         'exportBrep',
         'exportBrepToString',
         'exportIges',
         'exportStep',  # converted
         'exportStl',
         'extrude',  # converted
         'findPlane',
         'fix',
         'fixTolerance',
         'fixWire',
         'fuse',
         'generalFuse',
         'getAllDerivedFrom',
         'getElement',
         'getFacesFromSubelement',
         'getTolerance',
         'globalTolerance',
         'hashCode',
         'importBinary',
         'importBrep',
         'importBrepFromString',
         'inTolerance',
         'isClosed',  # converted
         'isCoplanar',
         'isDerivedFrom',
         'isEqual',
         'isInfinite',
         'isInside',
         'isNull',  # converted
         'isPartner',
         'isSame',
         'isValid',
         'limitTolerance',
         'makeChamfer',
         'makeFillet',
         'makeHomogenousWires',
         'makeOffset',
         'makeOffset2D',
         'makeOffsetShape',
         'makeParallelProjection',
         'makePerspectiveProjection',
         'makePipe',
         'makePipeShell',
         'makeShapeFromMesh',
         'makeThickness',
         'makeWires',
         'mirror',
         'multiFuse',
         'nullify',
         'oldFuse',
         'optimalBoundingBox',
         'overTolerance',
         'project',
         'proximity',
         'read',
         'reflectLines',
         'removeInternalWires',
         'removeShape',
         'removeSplitter',
         'replaceShape',
         'restoreContent',
         'reverse',
         'reversed',
         'revolve',  # converted
         'rotate',
         'rotated',
         'scale',  # converted
         'scaled',
         'section',
         'sewShape',
         'slice',
         'slices',
         'tessellate',
         'toNurbs',
         'transformGeometry',
         'transformShape',
         'transformed',
         'translate',
         'translated',
         'writeInventor']
