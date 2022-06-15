# import data
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import cmocean
import colorcet as cc
import scipy.interpolate as interper
from scipy.io import loadmat
from tqdm.notebook import tqdm as loading
import scipy.integrate as integrator
from matplotlib.colors import ListedColormap
import pyvista
import osnap_tools as ost

import scipy.signal as sig

##
path = (
    "/Users/manishdevana/Research/data_storage/Multibeam/"  # update for your filepath
)
topo = loadmat(path + "osnap_merged_bathy_ss_french_knorr.mat")
lon = topo["longrid"].flatten()
lat = topo["latgrid"].flatten()
topo = topo["merged_bathy_ss_french_knorr"]
topoOsnap = ost.loadTopo()

lons = topoOsnap["moor_lon"].flatten()
lats = topoOsnap["moor_lat"].flatten()

grid = pyvista.UniformGrid()
grid.dimensions = (lon.shape[0], lat.shape[0], 1)
topo2 = topo
grid.point_data["topo"] = topo2.flatten("C")
grid = grid.warp_by_scalar(
    factor=0.2
)  # .2 is somewhat arbitray but makes it visually appealing
# make into a draped 3d mesh
z_cells = np.array(
    [25] * 5 + [35] * 3 + [50] * 2 + [75, 100]
)  # see pyvista for explanation of this warping factor

xx = np.repeat(grid.x, len(z_cells), axis=-1)
yy = np.repeat(grid.y, len(z_cells), axis=-1)
zz = np.repeat(grid.z, len(z_cells), axis=-1) - np.cumsum(z_cells).reshape((1, 1, -1))

mesh = pyvista.StructuredGrid(xx, yy, zz)
mesh["Elevation"] = zz.ravel(order="F")


##
topoOsnap = ost.loadTopo()
lons = topoOsnap["moor_lon"].flatten()
lats = topoOsnap["moor_lat"].flatten()

Flon = interper.interp1d(lon, np.arange(lon.shape[0]))
Flat = interper.interp1d(lat, np.arange(lat.shape[0]),)
# for each trace need an x a y a nd z

points = [[Flon(lonin), Flat(latin), -100] for lonin, latin in zip(lons, lats)]

path = np.vstack(points)

ntraces = 1000
zfake = []
for zz in np.linspace(-3200, 0, ntraces):
    zz = np.full((lons.shape[0]), zz)
    zfake.append(zz)

zfake = np.stack(zfake).T


nsamples = zfake.shape[1]
# n
zspacing = 0.12
path = pyvista.wrap(path).points
points2 = np.repeat(path, nsamples, axis=0)
# nsamples, ntraces = topo.shape

# repeat the Z locations across
z_spacing = 0.1
tp = np.arange(0, z_spacing * nsamples, z_spacing)
tp = path[:, 2][:, None] - tp

points2[:, -1] = tp.ravel()
points2 = pyvista.wrap(points2).points

# grid2 = pv.StructuredGrid(points2[:,0], points2[:, 1], points2[:, 2])
grid2 = pyvista.StructuredGrid()
grid2.points = points2

grid2.dimensions = nsamples, 9, 1

grid2
# # Add the data array - note the ordering!
grid2["values"] = zfake.ravel(order="C")
# grid2['elev'] = pyvista.plotting.normalize(grid['e']) * 100
grid2 = grid2.warp_by_scalar(factor=0.2)

topoOsnap = ost.loadTopo()
lons = topoOsnap["moor_lon"].flatten()
lats = topoOsnap["moor_lat"].flatten()

Flon = interper.interp1d(lon, np.arange(lon.shape[0]))
Flat = interper.interp1d(lat, np.arange(lat.shape[0]),)

# for each trace need an x a y a nd z
points = [[Flon(lonin), Flat(latin), -200] for lonin, latin in zip(lons, lats)]

path = np.vstack(points)

ntraces = 1000
zfake = []
for zz in np.linspace(-3200, 0, ntraces):
    zz = np.full((lons.shape[0]), zz)
    zfake.append(zz)

zfake = np.stack(zfake).T

# fake plane data
nsamples = zfake.shape[1]
# n
zspacing = 0.12
path = pyvista.wrap(path).points
points2 = np.repeat(path, nsamples, axis=0)


# repeat the Z locations across
z_spacing = 0.1
tp = np.arange(0, z_spacing * nsamples, z_spacing)
tp = path[:, 2][:, None] - tp

points2[:, -1] = tp.ravel()
points2 = pyvista.wrap(points2).points

size = [6000, 4000]
plotter = pyvista.Plotter(
    off_screen=True,
    # lighting='light_kit',
    window_size=size,
    # cpos=cpos,
    point_smoothing=True,
    polygon_smoothing=True,
    # multi_samples=8
)
plotter.background_color = "k"
plotter.eye_dome_lighting = True
plotter.add_mesh(
    grid,
    show_scalar_bar=False,
    # lighting=False,
    diffuse=0.5,
    #                  specular=0.5,
    ambient=0.5,
    cmap="nipy_spectral",
)
plotter.add_mesh(grid2, color="red", opacity=0.6)
# mesh.plot(
#     # show_edges=True,
#           lighting=True,
#     # cpos=cpos
# )
plotter.camera_position = [
    (1152.5904239115132, -1516.9928158608395, 2454.0124160254572),
    (667.5, 557.0, -420.9651679992676),
    (-0.12302234427330311, 0.79487271126025, 0.5941741122796246),
]
# grid2.plot(show_axes=True,color='r',  )
plotter.add_mesh(pyvista.PolyData(path), color="white", point_size=20)
# plotter.show(interactive=False)
plotter.show(interactive=False, screenshot="map.png")
# plotter.screenshot('Map3d.png')
# plotter.show()
