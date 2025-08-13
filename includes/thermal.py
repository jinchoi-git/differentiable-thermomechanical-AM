import jax
import jax.numpy as jnp
import numpy as np

# Thermal functions     
def update_birth(current_time, element_birth, node_birth, surface_birth):
    ae = (element_birth <= current_time)
    an = (node_birth <= current_time)
    asf = (surface_birth[:, 0] <= current_time) & (surface_birth[:, 1] > current_time)
    return ae.astype(jnp.float32), an.astype(jnp.float32), asf.astype(jnp.float32)

def clear_vectors(num_nodes):
    m_vec = jnp.zeros(num_nodes)
    rhs = jnp.zeros(num_nodes)
    return m_vec, rhs             

def calc_cp(element, temperature, Nip_ele, solidus, liquidus, cp_val, latent):
    temperature_nodes = temperature[element]
    theta_ip = jnp.matmul(Nip_ele, temperature_nodes)

    k = 0.5  # 1/K sharpness, tune
    s0 = jax.nn.sigmoid((theta_ip - solidus) * k)
    s1 = jax.nn.sigmoid((liquidus - theta_ip) * k)
    cp_el = cp_val + latent * (s0 * s1)
    return cp_el
    
def calc_mass_stiff(element, cp, cond, temperature, nodes, Bip_ele, Nip_ele, density):
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
    
def update_mvec_stiffness(m_vec, rhs, cps, conds, elements, temperature, active_element_inds, active_node_inds, nodes, Bip_ele, Nip_ele, density):
    lump_mass, stiff_temp = jax.vmap(calc_mass_stiff, in_axes=(0, 0, 0, None, None, None, None, None))(elements, cps, conds, temperature, nodes, Bip_ele, Nip_ele, density)    
    lump_mass = lump_mass * active_element_inds[:,None]
    stiff_temp = stiff_temp * active_element_inds[:,None]
    m_vec = m_vec.at[elements].add(lump_mass)
    rhs = rhs.at[elements].add(-stiff_temp)
    return m_vec, rhs

def calc_flux(surface, surf_detJac, surface_xy, surface_flux, t, temperature, base_power, control,
              Nip_sur, nodes, laser_loc, laser_ont, r_beam, h_conv, ambient, h_rad):
    ip_pos = jnp.matmul(Nip_sur, nodes[surface])
    r2 = jnp.square(jnp.linalg.norm(ip_pos - laser_loc[jnp.newaxis, t].repeat(4, axis=0), axis=1))
    qmov = 2.0 * base_power * laser_ont * control / (3.14 * jnp.square(r_beam)) * jnp.exp(-2.0 * r2 / (jnp.square(r_beam))) * surface_xy
    temperature_nodes = temperature[surface]
    temperature_ip = jnp.matmul(Nip_sur, temperature_nodes)
    qconv = -1 * h_conv * (temperature_ip - ambient)
    qrad = -1 * 5.67e-14 * h_rad * (temperature_ip**4 - ambient**4)  
    q = ((qmov + qconv + qrad) * surf_detJac)@ Nip_sur * surface_flux
    return q

def update_fluxes(t, rhs, temperature, base_power, control, active_surface_inds,
                  surfaces, surf_detJacs, surface_xy, surface_flux, Nip_sur, nodes, laser_loc, laser_ont, r_beam, h_conv, ambient, h_rad):  
    flux_contribs = jax.vmap(
        calc_flux,
        in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None)
    )(
        surfaces, surf_detJacs, surface_xy, surface_flux, t, temperature, base_power, control,
        Nip_sur, nodes, laser_loc, laser_ont, r_beam, h_conv, ambient, h_rad
    )
    flux_contribs = flux_contribs * active_surface_inds[:,None]
    rhs = rhs.at[surfaces].add(flux_contribs)
    return rhs