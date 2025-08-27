import os
import time
import pyvista as pv
import vtk
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import time
import optax
import sys
from jax.scipy.sparse.linalg import cg
import jax.experimental.sparse as jsp  
from collections import namedtuple
from tqdm import trange
from functools import partial
import matplotlib.pyplot as plt

np.bool = np.bool_
os.environ["CUDA_VISIBLE_DEVICES"]="2"
output_dir = "./forward_MWE"
dir_path = './opt_MWE_reg'
scaler = 1.0
os.makedirs(output_dir, exist_ok=True)

'''
Load mesh information from .k file in LS-DYNA input format

Return node positions, node birth times, element nodes,
element birth times, surfaces, and surface birth times.
'''
def load_inputfile(filename='data/0.k'):
    nodes = []
    node_sets = {}
    elements = []
    birth_list_element = []
    birth_list_node = []

    with open(filename) as f:
        while True:
            line = next(f)
            if not line.split():
                continue
            if line.split()[0] == '*NODE':
                first = True
                while True:
                    line = next(f)
                    if line[0] == '*':
                        break
                    if line[0] == '$':
                        continue
                    text = line.split()
                    if first:
                        node_base = int(text[0])
                        first = False
                    nodes.append([float(text[1]),float(text[2]),float(text[3])])
            if line.split()[0] == '*END':
                break  
    birth_list_node = [-1 for _ in range(len(nodes))]
    with open(filename) as f:
        line = next(f)
        while True:
            if not line.split():
                line = next(f)
                continue
            elif line.split()[0] == '*SET_NODE_LIST':
                line = next(f)
                line = next(f)
                key = int(line.split()[0])
                node_list = []
                while True:
                    line = next(f)
                    if line[0] == '*':
                        break
                    if line[0] == '$':
                        continue
                    for text in line.split():
                        node_list.append(int(text)-node_base)
                node_sets[key] = node_list
            elif line.split()[0] == '*END':
                break
            else:
                line = next(f)
    with open(filename) as f:
        while True:
            line = next(f)
            if not line.split():
                continue
            if line.split()[0] == '*ELEMENT_SOLID':
                first = True
                while True:
                    line = next(f)
                    if line[0] == '*':
                        break
                    if line[0] == '$':
                        continue
                    text = line.split()
                    if first:
                        element_base = int(text[0])
                        first = False
                    elements.append([int(text[2])-node_base, int(text[3])-node_base, int(text[4])-node_base, int(text[5])-node_base,
                                     int(text[6])-node_base, int(text[7])-node_base, int(text[8])-node_base, int(text[9])-node_base])
            if line.split()[0] == '*END':
                break
    birth_list_element = [-1.0]*len(elements)
    with open(filename) as f:
        while True:
            line = next(f)
            if not line.split():
                continue
            if line.split()[0] == '*DEFINE_CURVE':
                while True:
                    line = next(f)
                    if line[0] == '*':
                        break
                    if line[0] == '$':
                        continue
                    text = line.split()
                    birth_list_element[int(float(text[1]))-element_base] = float(text[0])
            if line.split()[0] == '*END':
                break
    for element, birth_element in zip(elements, birth_list_element):
        if birth_element < 0:
            continue
        for node in element:
            if (birth_list_node[node] > birth_element or 
                                        birth_list_node[node] < 0):
                                    birth_list_node[node] = birth_element
    
    # create element surface for node ids
    element_surface = np.zeros((len(elements), 6, 4), dtype=np.int32)
    for ind, element in enumerate(elements):
        element_surface[ind, 0, :] = [element[4], element[5], element[6],element[7]]
        element_surface[ind, 1, :] = [element[0], element[1], element[2],element[3]]
        element_surface[ind, 2, :] = [element[0], element[1], element[5],element[4]]
        element_surface[ind, 3, :] = [element[3], element[2], element[6],element[7]]
        element_surface[ind, 4, :] = [element[0], element[3], element[7],element[4]]
        element_surface[ind, 5, :] = [element[1], element[2], element[6],element[5]]

    # create element surface neighbors 
    def match_surface(ind, s_ind):
        for surf_ind, surf in enumerate(element_surface[ind]):
            for s_surf in element_surface[s_ind]:
                if set(surf) == set(s_surf):
                    return surf_ind
        else:
            return -1

    element_surface_neighbor = np.ones((len(elements), 6), dtype=np.int32) * -1
    for ind, element in enumerate(elements):
        for s_ind, s_element in enumerate(elements):
            if any(node in element for node in s_element) and ind != s_ind:
                surf_ind = match_surface(ind, s_ind)
                if surf_ind != -1:
                    element_surface_neighbor[ind, surf_ind] = s_ind

    # create element surface birth
    element_surface_birth = np.zeros((len(elements), 6, 2), dtype=float)
    for ele_ind, element_surface_nei in enumerate(element_surface_neighbor):
        for sur_ind, neighbor_ind in enumerate(element_surface_nei):
            if neighbor_ind == -1:
                birth = birth_list_element[ele_ind]
                death = 1.0e6
            else:
                birth = min(birth_list_element[ele_ind], birth_list_element[neighbor_ind])
                death = max(birth_list_element[ele_ind], birth_list_element[neighbor_ind])
            element_surface_birth[ele_ind, sur_ind] = [birth, death]
            
    return nodes, birth_list_node, elements, birth_list_element, element_surface, element_surface_birth

def load_toolpath(filename = 'data/toolpath_c.crs', dt = 0.02):
    toolpath_raw=pd.read_table(filename, delimiter=r"\s+",header=None, names=['time','x','y','z','state'])
    toolpath=[]
    state=[]
    ctime=0.0
    ind=0
    endTime = float(toolpath_raw.tail(1)['time'])
    while(ctime<=endTime):
        while(ctime>=toolpath_raw['time'][ind+1]):
            ind=ind+1
        X=toolpath_raw['x'][ind]+(toolpath_raw['x'][ind+1]-toolpath_raw['x'][ind])*(
            ctime-toolpath_raw['time'][ind])/(toolpath_raw['time'][ind+1]-toolpath_raw['time'][ind])
        Y=toolpath_raw['y'][ind]+(toolpath_raw['y'][ind+1]-toolpath_raw['y'][ind])*(
            ctime-toolpath_raw['time'][ind])/(toolpath_raw['time'][ind+1]-toolpath_raw['time'][ind])
        Z=toolpath_raw['z'][ind]+(toolpath_raw['z'][ind+1]-toolpath_raw['z'][ind])*(
            ctime-toolpath_raw['time'][ind])/(toolpath_raw['time'][ind+1]-toolpath_raw['time'][ind])
        toolpath.append([X,Y,Z])
        state.append(toolpath_raw['state'][ind+1])
        ctime = ctime + dt
    return toolpath, state, endTime

def shape_fnc_element(parCoord):
    chsi = parCoord[0]
    eta = parCoord[1]
    zeta = parCoord[2]
    N =  0.125 * jnp.stack([(1.0 - chsi)*(1.0 - eta)*(1.0 - zeta),(1.0 + chsi)*(1.0 - eta)*(1.0 - zeta),
                           (1.0 + chsi)*(1.0 + eta)*(1.0 - zeta), (1.0 - chsi)*(1.0 + eta)*(1.0 - zeta),
                           (1.0 - chsi)*(1.0 - eta)*(1.0 + zeta), (1.0 + chsi)*(1.0 - eta)*(1.0 + zeta),
                           (1.0 + chsi)*(1.0 + eta)*(1.0 + zeta), (1.0 - chsi)*(1.0 + eta)*(1.0 + zeta)])
    return N
    
def derivate_shape_fnc_element(parCoord):
    oneMinusChsi = 1.0 - parCoord[0]
    onePlusChsi  = 1.0 + parCoord[0]
    oneMinusEta  = 1.0 - parCoord[1]
    onePlusEta   = 1.0 + parCoord[1]
    oneMinusZeta = 1.0 - parCoord[2]
    onePlusZeta  = 1.0 + parCoord[2]
    B = 0.1250 * jnp.array([[-oneMinusEta * oneMinusZeta, oneMinusEta * oneMinusZeta, 
                                onePlusEta * oneMinusZeta, -onePlusEta * oneMinusZeta, 
                                -oneMinusEta * onePlusZeta, oneMinusEta * onePlusZeta, 
                                onePlusEta * onePlusZeta, -onePlusEta * onePlusZeta],
                              [-oneMinusChsi * oneMinusZeta, -onePlusChsi * oneMinusZeta, 
                               onePlusChsi * oneMinusZeta, oneMinusChsi * oneMinusZeta, 
                               -oneMinusChsi * onePlusZeta, -onePlusChsi * onePlusZeta, 
                               onePlusChsi * onePlusZeta, oneMinusChsi * onePlusZeta],
                               [-oneMinusChsi * oneMinusEta, -onePlusChsi * oneMinusEta, 
                                -onePlusChsi * onePlusEta, -oneMinusChsi * onePlusEta, 
                                oneMinusChsi * oneMinusEta, onePlusChsi * oneMinusEta, 
                                onePlusChsi * onePlusEta, oneMinusChsi * onePlusEta]])
    return B

def shape_fnc_surface(parCoord):
    N = jnp.zeros((4))
    chsi = parCoord[0]
    eta  = parCoord[1]
    N = 0.25 * jnp.array([(1-chsi)*(1-eta), (1+chsi)*(1-eta), (1+chsi)*(1+eta), (1-chsi)*(1+eta)])
    return N

def derivate_shape_fnc_surface(parCoord):
    oneMinusChsi = 1.0 - parCoord[0]
    onePlusChsi  = 1.0 + parCoord[0]
    oneMinusEta  = 1.0 - parCoord[1]
    onePlusEta   = 1.0 + parCoord[1]
    B = 0.25 * jnp.array([[-oneMinusEta, oneMinusEta, onePlusEta, -onePlusEta], 
                         [-oneMinusChsi, -onePlusChsi, onePlusChsi, oneMinusChsi]])
    return B

def surface_jacobian(nodes, surfaces):
    nodes_pos = nodes[surfaces]
    mapped_surf_nodes_pos = jnp.zeros([nodes_pos.shape[0],4,2])
    u = nodes_pos[:,1,:] - nodes_pos[:,0,:]
    v = nodes_pos[:,2,:] - nodes_pos[:,1,:]
    w = nodes_pos[:,3,:] - nodes_pos[:,0,:]
    l1 = jnp.linalg.norm(u,axis=1)
    l2 = jnp.linalg.norm(v,axis=1)
    l4 = jnp.linalg.norm(w,axis=1)
    cos12 = (u[:,0]*v[:,0] + u[:,1]*v[:,1] + u[:,2]*v[:,2])/(l1*l2)
    cos14 = (u[:,0]*w[:,0] + u[:,1]*w[:,1] + u[:,2]*w[:,2])/(l1*l4)
    sin12 = jnp.sqrt(1.0 - cos12*cos12)
    sin14 = jnp.sqrt(1.0 - cos14*cos14)
    mapped_surf_nodes_pos = jnp.zeros([nodes_pos.shape[0],4,2]).at[:,1,0].set(l1).at[:,2,0].set(l1 + l2*cos12) \
                                    .at[:,2,1].set(l2*sin12).at[:,3,0].set(l4*cos14).at[:,3,1].set(l4*sin14)

    Jac = jnp.matmul(Bip_sur, mapped_surf_nodes_pos[:,jnp.newaxis,:,:].repeat(4,axis=1))
    surf_detJac = jnp.linalg.det(Jac)
    return surf_detJac

Load_data = True # always True I think
dt = 0.01

data_dir = 'preprocessed_10x5'
toolpath_name = '10x5_toolpath.crs'

if Load_data:
    elements = jnp.load(f'{data_dir}/elements.npy')
    nodes = jnp.load(f'{data_dir}/nodes.npy')
    surfaces = jnp.load(f'{data_dir}/surface.npy')
    node_birth = jnp.load(f'{data_dir}/node_birth.npy')
    element_birth = jnp.load(f'{data_dir}/element_birth.npy')
    surface_birth = jnp.load(f'{data_dir}/surface_birth.npy')
    surface_xy = jnp.load(f'{data_dir}/surface_xy.npy')
    surface_flux = jnp.load(f'{data_dir}/surface_flux.npy')
    
toolpath, state, endTime = load_toolpath(filename=toolpath_name, dt = dt)
parCoords_element = jnp.array([[-1.0,-1.0,-1.0],[1.0,-1.0,-1.0],[1.0, 1.0,-1.0],[-1.0, 1.0,-1.0],
             [-1.0,-1.0,1.0],[1.0,-1.0, 1.0], [ 1.0,1.0,1.0],[-1.0, 1.0,1.0]]) * 0.5773502692
parCoords_surface = jnp.array([[-1.0,-1.0],[-1.0, 1.0],[1.0,-1.0],[1.0,1.0]])* 0.5773502692

Nip_ele = np.array([shape_fnc_element(parCoord) for parCoord in parCoords_element]) #[:,:,jnp.newaxis]
Bip_ele = np.array([derivate_shape_fnc_element(parCoord) for parCoord in parCoords_element])
Nip_sur = np.array([shape_fnc_surface(parCoord) for parCoord in parCoords_surface])
Bip_sur = np.array([derivate_shape_fnc_surface(parCoord) for parCoord in parCoords_surface])
surf_detJacs = surface_jacobian(nodes, surfaces)

print("Number of nodes: {}".format(len(nodes)))
print("Number of elements: {}".format(len(elements)))
print("Number of surfaces: {}".format(len(surfaces)))
print("Number of time-steps: {}".format(len(toolpath)))

toolpath_np = np.loadtxt(f'./{toolpath_name}')

# Thermal functions
def update_birth(real_time):
    ae = (element_birth <= real_time)
    an = (node_birth <= real_time)
    asf = (surface_birth[:, 0] <= real_time) & (surface_birth[:, 1] > real_time)
    return ae.astype(jnp.float32), an.astype(jnp.float32), asf.astype(jnp.float32)

def clear_vectors(num_nodes):
    m_vec = jnp.zeros(num_nodes)
    rhs = jnp.zeros(num_nodes)
    return m_vec, rhs             

def calc_cp(element, temperature):
    temperature_nodes = temperature[element]
    theta_ip = jnp.matmul(Nip_ele, temperature_nodes)
    cp_el = jnp.where((theta_ip > solidus) & (theta_ip < liquidus), cp_val + latent, cp_val)
    return cp_el
    
def calc_mass_stiff(element, cp, cond, temperature):
    nodes_pos = nodes[element] # (8, 3)
    temperature_nodes = temperature[element] # (8, )
    Jac = jnp.matmul(Bip_ele, nodes_pos) # (8, 3, 3)
    detJac = jnp.linalg.det(Jac) # (8, )

    iJac = jnp.linalg.inv(Jac) # (8, 3, 3)  
    gradN = jnp.matmul(iJac, Bip_ele) # (8,3,3) @ (8,3,8) -> (8,3,8)
    gradN_T = jnp.transpose(gradN, axes=(0,2,1)) # (8, 8, 3)

    mass = (density * cp * detJac)[:, jnp.newaxis] * jnp.sum(Nip_ele[:,:,jnp.newaxis]@Nip_ele[:,jnp.newaxis,:], axis= 0)
    lump_mass= jnp.sum(mass,axis=1)
    stiffness = jnp.sum((cond * detJac)[:, jnp.newaxis] * jnp.matmul(gradN_T, gradN), axis=0)
    stiff_temp = jnp.matmul(stiffness, temperature_nodes)
    return lump_mass, stiff_temp
    
def update_mvec_stiffness(m_vec, rhs, cps, elements, temperature, active_element_inds, active_node_inds):
    lump_mass, stiff_temp = jax.vmap(calc_mass_stiff, in_axes=(0, 0, 0, None))(elements, cps, conds_init, temperature)    
    lump_mass = lump_mass * active_element_inds[:,None]
    stiff_temp = stiff_temp * active_element_inds[:,None]
    m_vec = m_vec.at[elements].add(lump_mass)
    rhs = rhs.at[elements].add(-stiff_temp)
    return m_vec, rhs

def calc_flux(surface, surf_detJac, surface_xy, surface_flux, t, temperature, base_power, control):
    ip_pos = jnp.matmul(Nip_sur, nodes[surface])
    r2 = jnp.square(jnp.linalg.norm(ip_pos - laser_loc[jnp.newaxis, t].repeat(4, axis=0), axis=1))
    qmov = 2.0 * base_power * laser_on[t] * control / (3.14 * jnp.square(r_beam)) * jnp.exp(-2.0 * r2 / (jnp.square(r_beam))) * surface_xy
    temperature_nodes = temperature[surface]
    temperature_ip = jnp.matmul(Nip_sur, temperature_nodes)
    qconv = -1 * h_conv * (temperature_ip - ambient)
    qrad = -1 * 5.67e-14 * h_rad * (temperature_ip**4 - ambient**4)  
    q = ((qmov + qconv + qrad) * surf_detJac)@ Nip_sur * surface_flux
    return q

def update_fluxes(t, rhs, temperature, base_power, control, active_surface_inds):  
    flux_contribs = jax.vmap(calc_flux, in_axes=(0, 0, 0, 0, None, None, None, None))(surfaces, surf_detJacs, surface_xy, surface_flux, t, temperature, base_power, control)
    flux_contribs = flux_contribs * active_surface_inds[:,None]
    rhs = rhs.at[surfaces].add(flux_contribs)
    return rhs

# Mech functions
def elastic_stiff_matrix(element, nodes, Bip_ele, shear, bulk):
    n_p = 8
    n_q = 8
    nodes_pos = nodes[element]
    Jac = jnp.matmul(Bip_ele, nodes_pos)
    ele_detJacs = jnp.linalg.det(Jac)

    iJac = jnp.linalg.inv(Jac)
    ele_gradN = jnp.matmul(iJac, Bip_ele)

    ele_B = jnp.zeros([n_q, 6, n_p * 3])
    ele_B = ele_B.at[:, 0, 0:24:3].set(ele_gradN[:, 0, :])
    ele_B = ele_B.at[:, 1, 1:24:3].set(ele_gradN[:, 1, :])
    ele_B = ele_B.at[:, 2, 2:24:3].set(ele_gradN[:, 2, :])
    ele_B = ele_B.at[:, 3, 0:24:3].set(ele_gradN[:, 1, :])
    ele_B = ele_B.at[:, 3, 1:24:3].set(ele_gradN[:, 0, :])
    ele_B = ele_B.at[:, 4, 1:24:3].set(ele_gradN[:, 2, :])
    ele_B = ele_B.at[:, 4, 2:24:3].set(ele_gradN[:, 1, :])
    ele_B = ele_B.at[:, 5, 2:24:3].set(ele_gradN[:, 0, :])
    ele_B = ele_B.at[:, 5, 0:24:3].set(ele_gradN[:, 2, :])

    IOTA = jnp.array([[1], [1], [1], [0], [0], [0]])
    VOL = jnp.matmul(IOTA, IOTA.T)
    DEV = jnp.diag(jnp.array([1, 1, 1, 1/2, 1/2, 1/2])) - VOL / 3
    ELASTC = 2 * DEV * shear[:, jnp.newaxis, jnp.newaxis] + VOL * bulk[:, jnp.newaxis, jnp.newaxis]
    
    ele_D = ele_detJacs[:, jnp.newaxis, jnp.newaxis] * ELASTC
    ele_K = jnp.matmul(jnp.matmul(ele_B.transpose([0, 2, 1]), ele_D), ele_B)
    ele_K = ele_K.sum(axis=0)

    return ele_K, ele_B, ele_D, ele_detJacs

def constitutive_problem(E, Ep_prev, Hard_prev, shear, bulk, a, Y, T_anneal=None, T=None):
    # Return-mapping algorithm with isotropic hardening.
    # 1. Setup volumetric & deviatoric projectors
    IOTA = jnp.array([[1], [1], [1], [0], [0], [0]])
    VOL = IOTA @ IOTA.T
    DEV = jnp.diag(jnp.array([1, 1, 1, 0.5, 0.5, 0.5])) - VOL / 3.0

    # 2. Elastic trial state
    E_tr = E - Ep_prev  # Elastic trial strain
    ELASTC = 2 * DEV * shear[..., None, None] + VOL * bulk[..., None, None]
    S_tr = jnp.squeeze(ELASTC @ E_tr[..., None])         # Full trial stress
    SD_tr = (2 * DEV * shear[..., None, None] @ E_tr[..., None]).squeeze() - Hard_prev  # Deviatoric stress minus backstress

    # -# 3. Yield criterion (von Mises style)
    sq = (
        jnp.sum(SD_tr[..., :3]**2, axis=-1)
      + 2 * jnp.sum(SD_tr[..., 3:]**2, axis=-1)
    )
    eps_norm = 1e-8
    norm_SD = jnp.sqrt(jnp.clip(sq, eps_norm, None))
    CRIT = norm_SD - Y
    IND_p = CRIT > 0.0    # Plastic indicator
    mask = IND_p.astype(jnp.float32)

    # 4. Safe computation (avoid divide-by-zero)
    safe_norm_SD = jnp.where(IND_p, norm_SD, 1.0)
    safe_denom = jnp.where(IND_p, 2 * shear + a, 1.0)
    N_hat  = SD_tr / safe_norm_SD[..., None]
    Lambda = jnp.where(IND_p, CRIT / safe_denom, 0.0)
    
    # 5. Stress update
    S = S_tr - 2 * shear[..., None] * Lambda[..., None] * N_hat

    # 6. Tangent modulus
    const = 4 * shear**2 / safe_denom
    term = const * Y / safe_norm_SD
    NN_hat = N_hat[..., :, None] @ N_hat[..., None, :]
    DS_plastic = -const[..., None, None] * DEV + term[..., None, None] * (DEV - NN_hat)
    DS = ELASTC + mask[..., None, None] * DS_plastic

    # 7. Plastic strain & hardening update
    Lambda_mat = Lambda[..., None] * jnp.array([1, 1, 1, 2, 2, 2])
    Ep = Ep_prev + Lambda_mat * N_hat
    Hard = Hard_prev + (a * Lambda)[..., None] * N_hat  # Isotropic hardening

    return S, DS, IND_p, Ep, Hard

def compute_E(element, ele_B, U_it):
    U_it = U_it[element]
    U_it_flattened = U_it.flatten()  # Shape: (24)
    E = jnp.squeeze(jnp.matmul(ele_B, U_it_flattened))
    return E

def transformation(Q_int, active_elements, ele_detJac, n_n_save):
    Q_int = Q_int.reshape(1, -1)
    elem = active_elements.transpose()  # elements.transpose() with shape (n_p=8, n_e)
    weight = ele_detJac.reshape(1, -1)
    n_e = elem.shape[1]  # number of elements
    n_p = 8  # number of vertices per element
    n_q = 8  # number of quadrature points
    n_int = n_e * n_q  # total number of integration points
    
    # values at integration points, shape(vF1)=shape(vF2)=(n_p, n_int)
    vF1 = jnp.ones((n_p, 1)) @ (weight * Q_int)
    vF2 = jnp.ones((n_p, 1)) @ weight

    # row and column indices, shape(iF)=shape(jF)=(n_p, n_int)
    iF = jnp.zeros((n_p, n_int), dtype=jnp.int32)
    jF = jnp.kron(elem, jnp.ones((1, n_q), dtype=jnp.int32))

    # Assemble using dense matrices for simplicity
    F1 = jnp.zeros((n_int, n_n_save))
    F2 = jnp.zeros((n_int, n_n_save))
    F1 = F1.at[(iF.flatten(), jF.flatten())].add(vF1.flatten())
    F2 = F2.at[(iF.flatten(), jF.flatten())].add(vF2.flatten())

    # Approximated values of the function Q at nodes of the FE mesh
    Q = F1 / F2
    Q_node = jnp.ones(Q.shape[1])
    Q_node = Q_node.at[0:n_n_save].set(Q[0, 0:n_n_save])
    
    return Q_node

def save_vtk(T_seq, S_seq, U_seq, elements, nodes, element_birth, node_birth, dt, out_dir="./vtk_out", keyword='') :
    os.makedirs(out_dir, exist_ok=True)
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
        filename = os.path.join(out_dir, f"{keyword}_{t:04d}.vtk")
        
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
        active_grid.point_data['temp'] = np.clip(np.array(T[0:n_n_save]), 300, 2300)
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

def mech(
    temperature,
    active_element_inds,
    active_node_inds,
    U,
    E,
    Ep_prev,
    Hard_prev,
    dU,
    current_time
):
    
    # Masks
    mask_e = active_element_inds                # (n_e,)
    mask_n = active_node_inds                   # (n_n,)
    n_n_active = jnp.sum(mask_n).astype(jnp.int32)
    n_dof = n_n * 3
    
    # If new layer is added, interpolate displacements for new nodes
    # U = disp_match(nodes, U, n_n_old)

    # # Interpolate temperature at integration points
    temperature_ip = (
        Nip_ele[:, jnp.newaxis, :] @ temperature[elements][:, jnp.newaxis, :, jnp.newaxis].repeat(8, axis=1)
    )[:, :, 0, 0]
    temperature_ip = jnp.clip(temperature_ip, 300, 2300)
       
    # Material properties
    young = jnp.interp(temperature_ip, temp_young1, young1)
    shear = young / (2 * (1 + poisson))
    bulk = young / (3 * (1 - 2 * poisson))
    scl = jnp.interp(temperature_ip, temp_scl1, scl1)
    Y = jnp.interp(temperature_ip, temp_Y1, Y1)
    a = a1 * jnp.ones_like(young)

    # Thermal strain
    alpha_Th = jnp.zeros((n_e, n_q, 6)).at[:, :, 0:3].set(scl[:, :, None].repeat(3, axis=2))
    E_th = (temperature_ip[:, :, None].repeat(6, axis=2) - T_Ref) * alpha_Th
    E_th = E_th * mask_e[:, None, None]
    
    # Elastic matrices (computed for all elements)
    ele_K, ele_B, ele_D, ele_detJac = jax.vmap(
        elastic_stiff_matrix, in_axes=(0, None, None, 0, 0)
    )(elements, nodes, Bip_ele, shear, bulk)

    # Zero out inactive elements
    ele_K      = ele_K * mask_e[:, None, None]
    ele_B      = ele_B * mask_e[:, None, None, None]
    ele_D      = ele_D * mask_e[:, None, None, None]
    ele_detJac = ele_detJac * mask_e[:, None]

    B_T = jnp.transpose(ele_B, (0, 1, 3, 2))  # (n_e, 8, 24, 6)
    
    # Dirichlet BC mask at DOF level
    Q_node = jnp.where(nodes[:, 2] < -2.9, 0.0, 1.0) * mask_n   # (n_n,)
    Q_dof  = jnp.repeat(Q_node, 3)                              # (n_dof,)

    def newton_iteration(i, state):
        U_it, dU = state

        # 1) Compute strain E and stress S at Gauss points
        E_base = jax.vmap(compute_E, in_axes=(0, 0, None))(elements, ele_B, U_it)
        E_corr = (E_base - E_th) * mask_e[:, None, None]
        S, DS, _, _, _ = constitutive_problem(E_corr, Ep_prev, Hard_prev, shear, bulk, a, Y)

        # 2) Tangent stiffness per element
        D_diff    = (ele_detJac[:, :, None, None] * DS) - ele_D
        B_T_D_B   = jnp.sum(B_T @ D_diff @ ele_B, axis=1)   # (n_e,24,24)
        K_tangent = ele_K + B_T_D_B                          # (n_e,24,24)
        
        # —— Stabilizer: for any element with mask_e==0, replace its 24×24 block by eps·I
        eps   = 1e-8
        eye24 = jnp.eye(24)[None, :, :]                       # shape (1,24,24)
        is_act = mask_e > 0.0                                 # boolean mask (n_e,)
        K_tangent = jnp.where(is_act[:, None, None],
                              K_tangent,                     # active: use true tangent
                              eps * eye24)                   # inactive: tiny identity

        # 3) Compute residual internal force F_node
        detS      = ele_detJac[..., None] * S               # (n_e,n_q,6)
        F_e       = jnp.einsum("eqik,eqk->ei", B_T, detS)    # (n_e,24)
        F_e       = F_e * mask_e[:, None]
        elem_dofs = jnp.repeat(elements * 3, 3, axis=1) + jnp.tile(jnp.arange(3), (n_e,8))
        F_node    = jnp.zeros((n_dof,))
        F_node    = F_node.at[elem_dofs.flatten()].add(F_e.flatten())

        # 4) Matrix‐free matvec for CG
        def mech_matvec(x):
            y0 = jnp.zeros_like(x)  # global accumulator
        
            def body(y_accum, e_idx):
                # 1) zero‐out inactive element
                Ke = K_tangent[e_idx] * mask_e[e_idx]      # (24×24)
                # Ke = K_tangent[e_idx]                        # (24×24)
                dofs = elem_dofs[e_idx]                    # (24,)
                local_x = x[dofs]                          # (24,)
                local_y = Ke @ local_x                     # (24,)
                # 2) scatter‐add
                y_accum = y_accum.at[dofs].add(local_y)
                return y_accum, None   # <-- now returns (carry, out)

            # Scan over elements:
            y, _ = jax.lax.scan(body, y0, jnp.arange(n_e))
        
            # Enforce Dirichlet rows & columns:
            return Q_dof * y * Q_dof  # shape (n_dof,)

        # 5) Solve for increment dU in flattened form
        resid     = -F_node * Q_dof
        # dU_flat, _ = cg(mech_matvec, resid, x0=jnp.zeros_like(resid), tol=cg_tol)
        
        dU_flat, info = cg(mech_matvec, resid, x0=jnp.zeros_like(resid), tol=cg_tol)
        
        # 6) Un-flatten and update
        dU_new = dU_flat.reshape((n_n, 3))
        U_it   = U_it + dU_new

        return U_it, dU
            
    state0 = (U * mask_n[:, None], jnp.zeros_like(U))
    U_it, dU = jax.lax.fori_loop(0, Maxit, newton_iteration, state0)

    # Final stress for output
    E_base = jax.vmap(compute_E, in_axes=(0, 0, None))(elements, ele_B, U_it)
    E_corr = (E_base - E_th) * mask_e[:, None, None]
    S_final, DS, IND_p, Ep_new, Hard_new = constitutive_problem(E_corr, Ep_prev, Hard_prev, shear, bulk, a, Y)
    
    # Update global U
    U = jax.lax.dynamic_update_slice(U, U_it, (0, 0))

    return (
        S_final,
        U,
        E,
        Ep_new,
        Hard_new,
        dU,
    )

# --- Thermal simulation ---
def simulate_temperature(control):
    def thermal_step(carry, t_control):
        temperature, temperatures = carry
        t, control_t = t_control
        current_time = t * dt
        ae, an, asurf = update_birth(current_time)

        cps = jax.vmap(calc_cp, in_axes=(0, None))(elements, temperature)
        m_vec = jnp.zeros(n_n)
        rhs = jnp.zeros(n_n)
        m_vec, rhs = update_mvec_stiffness(m_vec, rhs, cps, elements, temperature, ae, an)
        rhs = update_fluxes(t, rhs, temperature, base_power, control_t, asurf)

        # update = dt * rhs / (m_vec + 1e-8)* an    
        update = dt * rhs / m_vec * an 
        
        temperature = temperature + update
        temperature = temperature.at[bot_nodes].set(ambient)
        temperatures = temperatures.at[t].set(temperature)
                       
        return (temperature, temperatures), None

    t_seq = jnp.arange(1, steps)
    control_seq = control[:-1]
    carry = (
        jnp.full((n_n,), ambient),
        jnp.full((steps, n_n), ambient)
    )
    (temperature_final, temperatures), _ = jax.lax.scan(thermal_step, carry, (t_seq, control_seq))
    return temperatures

# --- Mechanical simulation ---
def simulate_mechanics(temperatures):
    def mech_scan_step(carry, t):
        state = carry
        T_t = temperatures[t]
        current_time = t * dt
        ae, an, _ = update_birth(current_time)

        S, U, E, Ep_new, Hard_new, dU = mech(
            T_t, ae, an, state.U, state.E, state.Ep_prev, state.Hard_prev,
            state.dU, current_time
        )
        new_state = MechState(U, E, Ep_new, Hard_new, dU)

        return new_state, (S, U)

    initial_mech_state = MechState(
        U=jnp.zeros((n_n, 3)),
        E=jnp.zeros((n_e, n_q, 6)),
        Ep_prev=jnp.zeros((n_e, n_q, 6)),
        Hard_prev=jnp.zeros((n_e, n_q, 6)),
        dU=jnp.zeros((n_n, 3)),
    )

    mech_timesteps = jnp.arange(1, steps, 10)
    final_state, (S_seq, U_seq) = jax.lax.scan(mech_scan_step, initial_mech_state, mech_timesteps)
    
    return S_seq, U_seq

def run_forward(input_control = None, output_path="./analysis", scaler=1.0):

    if not input_control:
        control_signal = jnp.ones((power_on_steps)) * scaler
        control = jnp.concatenate([
            control_signal,
            jnp.zeros((steps - power_on_steps,))
        ])
        keyword = "baseline"
        print("starting baseline forward")
    
    else:
        control = jnp.load(input_control)
        keyword = "optimized"
        print("starting optimized forward")
        
    # Run forward simulations
    temperatures = simulate_temperature(control)   
    S_seq, U_seq = simulate_mechanics(temperatures)
    save_vtk(temperatures, S_seq, U_seq, elements, nodes, element_birth, node_birth, dt, out_dir=output_path, keyword = keyword)
    print("finished")

# --- Simulation state containers ---
ThermalState = namedtuple("ThermalState", ["temperature", "temperatures"])
MechState = namedtuple("MechState", ["U", "E", "Ep_prev", "Hard_prev", "dU"])

# Time and mesh
steps = int(endTime / dt)
power_on_steps = 500
n_n = len(nodes)
n_e = len(elements)
n_p = 8
n_q = 8

# Material & heat transfer properties
ambient = 300.0
dt = 0.01
density = 0.0044
cp_val = 0.714
cond_val = 0.0178
Qin = 400.0 * 0.4
base_power = Qin
r_beam = 1.12
h_conv = 0.00005
h_rad = 0.2
solidus = 1878
liquidus = 1928
latent = 286 / (liquidus - solidus)
conds_init = jnp.ones((n_e, 8)) * cond_val

# Dirichlet boundary
bot_nodes = nodes[:, 2] < -2.9

# Laser activation
laser_loc = jnp.array(toolpath)
laser_on = jnp.array(state)
element_birth = jnp.array(element_birth)
node_birth = jnp.array(node_birth)

# Material models
poisson = 0.3
a1 = 10000
young1 = jnp.array(np.loadtxt('./materials/TI64_Young_Debroy.txt')[:, 1]) / 1e6
temp_young1 = jnp.array(np.loadtxt('./materials/TI64_Young_Debroy.txt')[:, 0])
Y1 = jnp.array(np.loadtxt('./materials/TI64_Yield_Debroy.txt')[:, 1]) / 1e6 * jnp.sqrt(2/3)
temp_Y1 = jnp.array(np.loadtxt('./materials/TI64_Yield_Debroy.txt')[:, 0])
scl1 = jnp.array(np.loadtxt('./materials/TI64_Alpha_Debroy.txt')[:, 1])
temp_scl1 = jnp.array(np.loadtxt('./materials/TI64_Alpha_Debroy.txt')[:, 0])
T_Ref = ambient

# Newton and CG tolerances
tol = 1e-4
cg_tol = 1e-4
# Maxit = 20
Maxit = 3

# Path to your training outputs
print(f"Looking in directory: {dir_path}")

# Assume control is shape (n_steps,)
n_steps = 500

# # Generate target sine wave baseline
# x = np.linspace(0, 2 * np.pi, n_steps)
# baseline = 1.0 + 0.25 * np.sin(x)  # oscillates between 0.75 and 1.25

# Helper to find the highest iteration number from filenames like 'loss_0005.npy'
def find_latest(prefix):
    files = [f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith('.npy')]
    print(f"Found {len(files)} '{prefix}_*.npy' files")
    if not files:
        raise FileNotFoundError(f"No files with prefix '{prefix}' in {dir_path}")
    iters = [int(f.split('_')[1].split('.')[0]) for f in files]
    latest_iter = max(iters)
    latest_file = f"{prefix}_{latest_iter:04d}.npy"
    print(f"→ Latest {prefix} file: {latest_file} (iter {latest_iter})")
    return os.path.join(dir_path, latest_file), latest_iter

# Load latest loss history
loss_path, loss_iter = find_latest('loss')
print(f"Loading loss history from: {loss_path}")
loss = np.load(loss_path)
print(f"Loaded loss array of shape {loss.shape}")

# Plot the loss history
print("Plotting loss history...")
plt.figure(figsize=(6,4), dpi=300)
plt.plot(loss, marker='o', markersize=3, linewidth=1)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(output_dir, "loss_history.png"))
print("Done plotting loss.")

# Collect all control files and sort by iteration number
control_files = [f for f in os.listdir(dir_path) if f.startswith('control_') and f.endswith('.npy')]
control_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))

# Load all control arrays
controls = [np.load(os.path.join(dir_path, f)) for f in control_files]
iters = [int(f.split('_')[1].split('.')[0]) for f in control_files]

# Plot every 10th control
print("Plotting control history...")
plt.figure(figsize=(6, 4), dpi=300)
for i, (ctrl, it) in enumerate(zip(controls, iters)):
    if i % 5 == 0 or i == loss_iter:
        plt.plot(ctrl[0:500], label=f'Iter {it}', linewidth=1)

# plt.plot(baseline, '--', color='gray', label='Target')
plt.xlabel('Timestep')
plt.ylabel('Normalized Control')
plt.legend(ncol=3, fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(output_dir, "control_history.png"))
print("Done plotting control.")

if __name__ == "__main__":
    run_forward(input_control = None, output_path = output_dir, scaler=scaler)
    run_forward(input_control = f"{dir_path}/control_0{loss_iter}.npy", output_path = output_dir, scaler=scaler)
    #run_forward(input_control = f"{dir_path}/control_0300.npy", output_path = output_dir)
