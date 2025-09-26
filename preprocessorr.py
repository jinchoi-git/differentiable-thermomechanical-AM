import sys, os
sys.path.append('/home/jyc3887/AM_Thermomechanical_Solver/includes')

# >>> Headless-safe: set before importing pyvista/vtk
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("PYVISTA_USE_PANEL", "false")

from preprocessor import write_keywords, write_birth, write_parameters
from gamma import domain_mgr, heat_solve_mgr, load_toolpath, get_toolpath
import cupy as cp
import numpy as np
import pyvista as pv
import vtk

# start virtual framebuffer + force offscreen
pv.start_xvfb()

cp.cuda.Device(0).use()
np.bool = np.bool_

file_name = '1x5.inp'  # input mesh file from abaqus
toolpath_file = 'toolpath_1x5.crs'
output_file = '1x5.k'  # define keyword file name

substrate_height = 1.
radius = 1.12
path_resolution = 0.1  # half of the element size

write_keywords(file_name, output_file, substrate_height)
write_birth(
    output_file,
    toolpath_file,
    path_resolution,
    radius,
    gif_end=2,
    nFrame=2,
    mode=1,
    camera_position=[(0, -50, 75), (0, 0, 0), (0.0, 0.0, 1.0)]
)
write_parameters(output_file)

domain = domain_mgr(filename='1x5.k')
heat_solver = heat_solve_mgr(domain)

data_dir = "./preprocessed_1x5"
os.makedirs(data_dir, exist_ok=True)
np.save(os.path.join(data_dir, 'elements'), domain.elements)
np.save(os.path.join(data_dir, 'nodes'), domain.nodes)          # fixed: remove leading dot
np.save(os.path.join(data_dir, 'surface'), domain.surface)
np.save(os.path.join(data_dir, 'node_birth'), domain.node_birth)
np.save(os.path.join(data_dir, 'element_birth'), domain.element_birth)
np.save(os.path.join(data_dir, 'surface_birth'), domain.surface_birth)
np.save(os.path.join(data_dir, 'surface_xy'), domain.surface_xy)
np.save(os.path.join(data_dir, 'surface_flux'), domain.surface_flux)
