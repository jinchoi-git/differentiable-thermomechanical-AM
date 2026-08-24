#!/usr/bin/env python
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_ROOT = PROJECT_ROOT / "preprocessed"
sys.path.append(str(PROJECT_ROOT / "includes"))
from preprocessor import (
            domain_mgr,
            heat_solve_mgr,
            write_birth,
            write_keywords,
            write_parameters,
)

np.bool = np.bool_

base_name = '3_mtsl'
file_name = PREPROCESSED_ROOT / f'{base_name}.inp' #input mesh file from abaqus
toolpath_file = PREPROCESSED_ROOT / f'{base_name}.crs'
output_file = PREPROCESSED_ROOT / f'{base_name}.k' #define keyword file name

with open(toolpath_file) as _toolpath_file:
    end_time = float(_toolpath_file.read().strip().splitlines()[-1].split()[0])
print(f"End time from toolpath file: {end_time}")
substrate_height = 1.
radius = 1.12
path_resolution = 0.1 # half of the element size
write_keywords(file_name, output_file, substrate_height)
write_birth(output_file, toolpath_file, path_resolution, radius, gif_end=0, nFrame=0, mode=1,
            camera_position=[(0, -50, 75), (0, 0, 0), (0.0, 0.0, 1.0)])
write_parameters(output_file, base_name=base_name, end_time=end_time)

domain = domain_mgr(filename=output_file)
heat_solver = heat_solve_mgr(domain)

data_dir = PREPROCESSED_ROOT / f"{base_name}_preprocessed"
os.makedirs(data_dir, exist_ok=True)
np.save(data_dir / 'elements', domain.elements)
np.save(data_dir / 'nodes', domain.nodes)          # fixed: remove leading dot
np.save(data_dir / 'surface', domain.surface)
np.save(data_dir / 'node_birth', domain.node_birth)
np.save(data_dir / 'element_birth', domain.element_birth)
np.save(data_dir / 'surface_birth', domain.surface_birth)
np.save(data_dir / 'surface_xy', domain.surface_xy)
np.save(data_dir / 'surface_flux', domain.surface_flux)
