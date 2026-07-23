from discretize import TensorMesh
from discretize.utils import active_from_xyz
from simpeg.utils import mkvc, model_builder
from simpeg import maps
import numpy as np
import matplotlib.pyplot as plt

# Defining the mesh

def make_example_mesh():
    dh = 5.0
    # Cell width, number of cells, expansion factor
    hx = [(dh, 5, -1.3), (dh, 20), (dh, 5, 1.3)]
    hy = [(dh, 5, -1.3), (dh, 20), (dh, 5, 1.3)]
    hz = [(dh, 5, -1.3), (dh, 20), (dh, 5, 1.3)]
    mesh = TensorMesh([hx, hy, hz], "CCC")

    return mesh

# Halfspace model with topography at z = 0


mesh = make_example_mesh()

halfspace_value = 100.0

air_value = 0.0
ind_active = mesh.gridCC[:, 2] < 0.0
model_map = maps.InjectActiveCells(mesh, ind_active, air_value)

model = halfspace_value * np.ones(ind_active.sum())

# Plot a slice of the model 
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ind_slice = int(mesh.shape_cells[1] / 2)
mesh.plot_slice(model_map * model, normal="Y", ax=ax, ind=ind_slice, grid=True)
ax.set_title("Model slice at y = {} m".format(mesh.cell_centers_y[ind_slice]))
plt.show()

# Topography, a block and a vertical dyke
mesh = make_example_mesh()

background_value = 100.0
dyke_value = 40.0
block_value = 70.0

[xx, yy] = np.meshgrid(mesh.nodes_x, mesh.nodes_y)
zz = -3 * np.exp((xx**2 + yy**2) / 75**2) + 40.0
topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

air_value = 0.0 
ind_active = active_from_xyz(mesh, topo, "N")
model_map = maps.InjectActiveCells(mesh, ind_active, air_value)

model = background_value * np.ones(ind_active.sum())
ind_dyke = (mesh.gridCC[ind_active, 0] > 20.0) & (mesh.gridCC[ind_active, 0] < 40.0)
model[ind_dyke] = dyke_value
ind_block = (
    (mesh.gridCC[ind_active, 0] > -40.0)
    & (mesh.gridCC[ind_active, 0] < -10.0)
    & (mesh.gridCC[ind_active, 1] > -30.0)
    & (mesh.gridCC[ind_active, 1] < 30.0)
    & (mesh.gridCC[ind_active, 2] > -40.0)
    & (mesh.gridCC[ind_active, 2] < 0.0)
)
model[ind_block] = block_value

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ind_slice = int(mesh.shape_cells[1] / 2)
mesh.plot_slice(model_map * model, normal="Y", ax=ax, ind=ind_slice, grid=True)
ax.set_title("Model slice at y = {} m".format(mesh.cell_centers_y[ind_slice]))
plt.show()

# Combo maps

mesh = make_example_mesh()

background_value = np.log(1.0 / 100.0)
dyke_value = np.log(1.0 / 40.0)
block_value = np.log(1.0 / 70.0)

[xx, yy] = np.meshgrid(mesh.nodes_x, mesh.nodes_y)
zz = -3 * np.exp((xx**2 + yy**2) / 75**2) + 40.0
topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

air_value = 0.0
ind_active = active_from_xyz(mesh, topo, "N")
active_map = maps.InjectActiveCells(mesh, ind_active, air_value)

model = background_value * np.ones(ind_active.sum())
ind_dyke = (mesh.gridCC[ind_active, 0] > 20.0) & (mesh.gridCC[ind_active, 0] < 40.0)
model[ind_dyke] = dyke_value
ind_block = (
    (mesh.gridCC[ind_active, 0] > -40.0)
    & (mesh.gridCC[ind_active, 0] < -10.0)
    & (mesh.gridCC[ind_active, 1] > -30.0)
    & (mesh.gridCC[ind_active, 1] < 30.0)
    & (mesh.gridCC[ind_active, 2] > -40.0)
    & (mesh.gridCC[ind_active, 2] < 0.0)
)
model[ind_block] = block_value

exponential_map = maps.ExpMap()
reciprocal_map = maps.ReciprocalMap()
model_map = active_map * reciprocal_map * exponential_map

# Plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ind_slice = int(mesh.shape_cells[1] / 2)
mesh.plot_slice(model_map * model, normal="Y", ax=ax, ind=ind_slice, grid=True)
ax.set_title("Model slice at y = {} m".format(mesh.cell_centers_y[ind_slice]))
plt.show()

# Models with arbitrary shapes

mesh = make_example_mesh()

background_value = 100.0
dyke_value = 40.0
sphere_value = 70.0

[xx, yy] = np.meshgrid(mesh.nodes_x, mesh.nodes_y)
zz = -3 * np.exp((xx**2 + yy**2) / 75**2) + 40.0
topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

air_value = 0.0
ind_active = active_from_xyz(mesh, topo, "N")
model_map = maps.InjectActiveCells(mesh, ind_active, air_value)

model = background_value * np.ones(ind_active.sum())

# Add  a sphere
ind_sphere = model_builder.get_indices_sphere(
    np.r_[-25.0, 0.0, -15.0], 20.0, mesh.gridCC
)
ind_sphere = ind_sphere[ind_active]
model[ind_sphere] = sphere_value

# Add a dyke
xp = np.kron(np.ones((2)), [-10.0, 10.0, 45.0, 25.0])
yp = np.kron([-1000.0, 1000.0], np.ones((4)))
zp = np.kron(np.ones((2)), [-120.0, -120.0, 35.0, 35.0])
xyz_pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
ind_polygon = model_builder.get_indices_polygon(mesh, xyz_pts)
ind_polygon = ind_polygon[ind_active]
model[ind_polygon] = dyke_value

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ind_slice = int(mesh.shape_cells[1] / 2)
mesh.plot_slice(model_map * model, normal="Y", ax=ax, ind=ind_slice, grid=True)
ax.set_title("Model slice at y = {} m".format(mesh.cell_centers_y[ind_slice]))
plt.show()

# Parametrized block model 
mesh = make_example_mesh()

background_value = 100.0  # background value
block_value = 40.0  # block value
xc, yc, zc = -25.0, 0.0, -20.0  # center of block
dx, dy, dz = 30.0, 40.0, 30.0  # dimensions in x,y,z

# Define surface topography
[xx, yy] = np.meshgrid(mesh.nodes_x, mesh.nodes_y)
zz = -3 * np.exp((xx**2 + yy**2) / 75**2) + 40.0
topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

# Set active cells and define unit values
air_value = 0.0
ind_active = active_from_xyz(mesh, topo, "N")
active_map = maps.InjectActiveCells(mesh, ind_active, air_value)

model = np.r_[background_value, block_value, xc, dx, yc, dy, zc, dz]
parametric_map = maps.ParametricBlock(
    mesh, active_cells=ind_active, epsilon=1e-10, p=5.0
)

# Define a single mapping from model to mesh
model_map = active_map * parametric_map

# Plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ind_slice = int(mesh.shape_cells[1] / 2)
mesh.plot_slice(model_map * model, normal="Y", ax=ax, ind=ind_slice, grid=True)
ax.set_title("Model slice at y = {} m".format(mesh.cell_centers_y[ind_slice]))
plt.show()


# Using wires map

mesh = make_example_mesh()

background_sigma = np.log(100.0)
sphere_sigma = np.log(70.0)
dyke_sigma = np.log(40.0)
background_myu = 1.0
sphere_mu = 1.25

# Define surface topography
[xx, yy] = np.meshgrid(mesh.nodes_x, mesh.nodes_y)
zz = -3 * np.exp((xx**2 + yy**2) / 75**2) + 40.0
topo = np.c_[mkvc(xx), mkvc(yy), mkvc(zz)]

# Set active cells
air_value = 0.0
ind_active = active_from_xyz(mesh, topo, "N")
active_map = maps.InjectActiveCells(mesh, ind_active, air_value)

# Define model for cells under the surface topography
N = int(ind_active.sum())
model = np.kron(np.ones((N, 1)), np.c_[background_sigma, background_myu])

# Add a conductive and permeable sphere
ind_sphere = model_builder.get_indices_sphere(
    np.r_[-25.0, 0.0, -15.0], 20.0, mesh.gridCC
)
ind_sphere = ind_sphere[ind_active]  # So same size and order as model
model[ind_sphere, :] = np.c_[sphere_sigma, sphere_mu]

# Add a conductive and non-permeable dyke
xp = np.kron(np.ones((2)), [-10.0, 10.0, 45.0, 25.0])
yp = np.kron([-1000.0, 1000.0], np.ones((4)))
zp = np.kron(np.ones((2)), [-120.0, -120.0, 35.0, 35.0])
xyz_pts = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
ind_polygon = model_builder.get_indices_polygon(mesh, xyz_pts)
ind_polygon = ind_polygon[ind_active]  # So same size and order as model
model[ind_polygon, 0] = dyke_sigma

# Create model vector and wires
model = mkvc(model)
wire_map = maps.Wires(("log_sigma", N), ("mu", N))

# Use combo maps to map from model to mesh
sigma_map = active_map * maps.ExpMap() * wire_map.log_sigma
mu_map = active_map * wire_map.mu

# Plot
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ind_slice = int(mesh.shape_cells[1] / 2)
mesh.plot_slice(sigma_map * model, normal="Y", ax=ax, ind=ind_slice, grid=True)
ax.set_title("Model slice at y = {} m".format(mesh.cell_centers_y[ind_slice]))
plt.show()


