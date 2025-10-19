#!/usr/bin/env python
# coding: utf-8
import os
import sys
import numpy as np
sys.path.append("./includes")
from preprocessor import write_keywords,write_birth,write_parameters, domain_mgr, heat_solve_mgr, load_toolpath, get_toolpath

np.bool = np.bool_

base_name = '4_mtml'
file_name = f'{base_name}.inp' #input mesh file from abaqus
toolpath_file = f'{base_name}.crs'
output_file = f'{base_name}.k' #define keyword file name

end_time = float(open(toolpath_file).read().strip().splitlines()[-1].split()[0])
print(f"End time from toolpath file: {end_time}")
substrate_height = 1.
radius = 1.12
path_resolution = 0.1 # half of the element size
write_keywords(file_name,output_file,substrate_height)
write_birth(output_file,toolpath_file,path_resolution,radius,gif_end=0,nFrame=0,mode=1,camera_position=[(0, -50, 75),(0, 0, 0),(0.0, 0.0, 1.0)])
write_parameters(output_file, base_name=base_name, end_time=end_time)

domain = domain_mgr(filename=f'{base_name}.k')
heat_solver = heat_solve_mgr(domain)

data_dir = f"./{base_name}_preprocessed"
os.makedirs(data_dir, exist_ok=True)
np.save(os.path.join(data_dir, 'elements'), domain.elements)
np.save(os.path.join(data_dir, 'nodes'), domain.nodes)          # fixed: remove leading dot
np.save(os.path.join(data_dir, 'surface'), domain.surface)
np.save(os.path.join(data_dir, 'node_birth'), domain.node_birth)
np.save(os.path.join(data_dir, 'element_birth'), domain.element_birth)
np.save(os.path.join(data_dir, 'surface_birth'), domain.surface_birth)
np.save(os.path.join(data_dir, 'surface_xy'), domain.surface_xy)
np.save(os.path.join(data_dir, 'surface_flux'), domain.surface_flux)