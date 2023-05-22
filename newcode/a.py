import sys

try:
    import gmsh
except ImportError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)


from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI

gmsh.initialize()

# Choose if Gmsh output is verbose
gmsh.option.setNumber("General.Terminal", 0)
model = gmsh.model()
model.add("Sphere")
model.setCurrent("Sphere")
sphere = model.occ.addSphere(0, 0, 0, 1, tag=1)

# Synchronize OpenCascade representation with gmsh model
model.occ.synchronize()

# Add physical marker for cells. It is important to call this function
# after OpenCascade synchronization
model.add_physical_group(3, [sphere])


# Generate the mesh
model.mesh.generate(3)

msh, cell_markers, facet_markers = gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
msh.name = "Sphere"
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

with XDMFFile(msh.comm, f"out_gmsh/mesh_rank_{MPI.COMM_WORLD.rank}.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_meshtags(cell_markers)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    file.write_meshtags(facet_markers)

model_rank = 0
mesh_comm = MPI.COMM_WORLD
if mesh_comm.rank == model_rank:
    # Generate
    model.add("Hexahedral mesh")
    model.setCurrent("Hexahedral mesh")

    # Recombine tetrahedrons to hexahedrons
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1)

    circle = model.occ.addDisk(0, 0, 0, 1, 1)
    circle_inner = model.occ.addDisk(0, 0, 0, 0.5, 0.5)
    cut = model.occ.cut([(2, circle)], [(2, circle_inner)])[0]
    extruded_geometry = model.occ.extrude(
        cut, 0, 0, 0.5, numElements=[5], recombine=True
    )
    model.occ.synchronize()

    model.addPhysicalGroup(2, [cut[0][1]], tag=1)
    model.setPhysicalName(2, 1, "2D cylinder")
    boundary_entities = model.getEntities(2)
    other_boundary_entities = []
    for entity in boundary_entities:
        if entity != cut[0][1]:
            other_boundary_entities.append(entity[1])
    model.addPhysicalGroup(2, other_boundary_entities, tag=3)
    model.setPhysicalName(2, 3, "Remaining boundaries")

    model.mesh.generate(3)
    model.mesh.setOrder(2)
    volume_entities = []
    for entity in extruded_geometry:
        if entity[0] == 3:
            volume_entities.append(entity[1])
    model.addPhysicalGroup(3, volume_entities, tag=1)
    model.setPhysicalName(3, 1, "Mesh volume")

msh, mt, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank)
msh.name = "hex_d2"
mt.name = f"{msh.name}_cells"
ft.name = f"{msh.name}_surface"


with XDMFFile(msh.comm, "out_gmsh/mesh.xdmf", "a") as file:
    file.write_mesh(msh)
    file.write_meshtags(
        mt, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
    )
    file.write_meshtags(
        ft, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
    )
