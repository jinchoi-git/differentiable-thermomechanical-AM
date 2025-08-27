import os
import numpy as np
import jax
import jax.numpy as jnp
import pyvista as pv
import vtk
from mech import transformation

def save_vtk(T_seq, S_seq, U_seq, elements, Bip_ele, nodes, element_birth, node_birth, dt, work_dir="./vtk_out", keyword="forward"):
    T_total = S_seq.shape[0]

    def get_detJacs(element):
        nodes_pos = nodes[element]
        Jac = jnp.matmul(Bip_ele, nodes_pos)
        ele_detJac_ = jnp.linalg.det(Jac)
        return ele_detJac_
    
    ele_detJac = jax.vmap(get_detJacs)(elements)
    
    for t in range(0, T_total):
        dt = 0.1 
        current_time = t * dt
        filename = os.path.join(work_dir, f"{keyword}_{t:04d}.vtk")
        
        S = S_seq[t]
        U = U_seq[t]
        T = T_seq[int(t*10)]   
        
        # Recompute activation masks at this time
        active_element_inds = np.array(element_birth <= current_time)
        active_node_inds = np.array(node_birth <= current_time)
        n_e_save = int(np.sum(active_element_inds))
        n_n_save = int(np.sum(active_node_inds))
        
        active_elements_list = elements[active_element_inds].tolist()
        active_cells = np.array([item for sublist in active_elements_list for item in [8] + sublist])
        active_cell_type = np.array([vtk.VTK_HEXAHEDRON] * len(active_elements_list))
        
        points = np.array(nodes[0:n_n_save] + U[0:n_n_save])
        # points = np.array(nodes[0:n_n_save])
        
        Sv = transformation(np.sqrt(1/2 * ((S[0:n_e_save,:,0] - S[0:n_e_save,:,1])**2 + 
                                        (S[0:n_e_save,:,1] - S[0:n_e_save,:,2])**2 + 
                                        (S[0:n_e_save,:,2] - S[0:n_e_save,:,0])**2 + 
                                        6 * (S[0:n_e_save,:,3]**2 + S[0:n_e_save,:,4]**2 + S[0:n_e_save,:,5]**2))), 
                        elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
        S11 = transformation(S[0:n_e_save,:,0], elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
        S22 = transformation(S[0:n_e_save,:,1], elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
        S33 = transformation(S[0:n_e_save,:,2], elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
        S12 = transformation(S[0:n_e_save,:,3], elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
        S23 = transformation(S[0:n_e_save,:,4], elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
        S13 = transformation(S[0:n_e_save,:,5], elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
    
        # Using pyvista for vtk
        active_grid = pv.UnstructuredGrid(active_cells, active_cell_type, points)
        # active_grid.point_data['temp'] = np.clip(np.array(T[0:n_n_save]), 300, 2300)
        active_grid.point_data['temp'] = np.array(T[0:n_n_save])
        active_grid.point_data['S_von'] = np.array(Sv)
        active_grid.point_data['S11'] = np.array(S11)
        active_grid.point_data['S22'] = np.array(S22)
        active_grid.point_data['S33'] = np.array(S33)
        active_grid.point_data['S12'] = np.array(S12)
        active_grid.point_data['S23'] = np.array(S23)
        active_grid.point_data['S13'] = np.array(S13)
        active_grid.point_data['U1'] = np.array(U[0:n_n_save, 0])
        active_grid.point_data['U2'] = np.array(U[0:n_n_save, 1])
        active_grid.point_data['U3'] = np.array(U[0:n_n_save, 2])
        active_grid.save(filename)

def find_latest(prefix, work_dir):
    files = [f for f in os.listdir(work_dir) if f.startswith(prefix) and f.endswith('.npy')]
    print(f"Found {len(files)} '{prefix}_*.npy' files")
    if not files:
        raise FileNotFoundError(f"No files with prefix '{prefix}' in {work_dir}")
    iters = [int(f.split('_')[1].split('.')[0]) for f in files]
    latest_iter = max(iters)
    latest_file = f"{prefix}_{latest_iter:04d}.npy"
    print(f"→ Latest {prefix} file: {latest_file} (iter {latest_iter})")
    return os.path.join(work_dir, latest_file), latest_iter