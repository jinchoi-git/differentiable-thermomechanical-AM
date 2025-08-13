import numpy as np
import pandas as pd
import jax.numpy as jnp

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
    endTime = float(toolpath_raw['time'].iloc[-1])
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

def surface_jacobian(nodes, surfaces, Bip_sur):
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