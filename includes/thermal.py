import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class ThermContext:
    # --- mesh / shapes ---
    n_n: int
    n_e: int
    n_q: int
    elements: jnp.ndarray
    nodes: jnp.ndarray
    Nip_ele: jnp.ndarray
    Bip_ele: jnp.ndarray

    # --- surfaces (keep if your flux assembly uses them; otherwise you can drop) ---
    surfaces: jnp.ndarray
    Nip_sur: jnp.ndarray
    surf_detJacs: jnp.ndarray
    surface_xy: jnp.ndarray
    surface_flux: jnp.ndarray

    # --- constant thermal properties & BC params ---
    ambient: float          # 300.0
    density: float          # 0.0044
    cp_val: float           # 0.714
    cond_val: float         # 0.01780  (scalar)
    conds: jnp.ndarray      # per-element/IP conductivity matrix (n_e, 8) * cond_val

    h_conv: float           # 0.00005
    emissivity: float       # 0.2
    stefan_boltz: float     # 5.670374419e-8 (if you use radiation anywhere)

    # --- phase window (if used in your model) ---
    solidus: float          # 1878
    liquidus: float         # 1928
    latent: float           # 286 / (liquidus - solidus)

    # --- laser & schedule ---
    base_power: float       # Qin (already includes absorptivity)
    r_beam: float           # 1.12
    laser_loc: jnp.ndarray  # (steps, 2|3) toolpath positions per step
    laser_on: jnp.ndarray   # (steps,) 0/1 on/off mask per step

    element_birth : jnp.ndarray
    node_birth    : jnp.ndarray
    surface_birth : jnp.ndarray

    # --- time / loop ---
    dt: float
    steps: int

    # --- boundary nodes mask (if used for Dirichlet at bottom) ---
    BOT_NODES: jnp.ndarray

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

    k = 1.0  # 1/K sharpness, tune
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
              Nip_sur, nodes, laser_loc, laser_ont, r_beam, h_conv, ambient, emissivity):
    ip_pos = jnp.matmul(Nip_sur, nodes[surface])
    r2 = jnp.square(jnp.linalg.norm(ip_pos - laser_loc[jnp.newaxis, t].repeat(4, axis=0), axis=1))
    qmov = 2.0 * base_power * laser_ont * control / (3.14 * jnp.square(r_beam)) * jnp.exp(-2.0 * r2 / (jnp.square(r_beam))) * surface_xy
    temperature_nodes = temperature[surface]
    temperature_ip = jnp.matmul(Nip_sur, temperature_nodes)
    qconv = -1 * h_conv * (temperature_ip - ambient)
    qrad = -1 * 5.67e-14 * emissivity * (temperature_ip**4 - ambient**4)  
    q = ((qmov + qconv + qrad) * surf_detJac)@ Nip_sur * surface_flux
    return q

def update_fluxes(t, rhs, temperature, base_power, control, active_surface_inds,
                  surfaces, surf_detJacs, surface_xy, surface_flux, Nip_sur, nodes, laser_loc, laser_ont, r_beam, h_conv, ambient, emissivity):  
    flux_contribs = jax.vmap(
        calc_flux,
        in_axes=(0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None)
    )(
        surfaces, surf_detJacs, surface_xy, surface_flux, t, temperature, base_power, control,
        Nip_sur, nodes, laser_loc, laser_ont, r_beam, h_conv, ambient, emissivity
    )
    flux_contribs = flux_contribs * active_surface_inds[:,None]
    rhs = rhs.at[surfaces].add(flux_contribs)
    return rhs

# --- Thermal simulation ---
def simulate_temperature(control, tctx):

     # ---- pull from context (no behavior change) ----
    n_n        = tctx.n_n
    n_e        = tctx.n_e
    n_q        = tctx.n_q
    elements   = tctx.elements
    nodes      = tctx.nodes
    Nip_ele    = tctx.Nip_ele
    Bip_ele    = tctx.Bip_ele

    surfaces      = tctx.surfaces
    Nip_sur       = tctx.Nip_sur
    surf_detJacs  = tctx.surf_detJacs
    surface_xy    = tctx.surface_xy
    surface_flux  = tctx.surface_flux

    ambient    = tctx.ambient
    density    = tctx.density
    cp_val     = tctx.cp_val
    cond_val   = tctx.cond_val
    conds      = tctx.conds  # preferred if you assemble per-element/IP k

    h_conv     = tctx.h_conv
    emissivity = tctx.emissivity
    sigma      = tctx.stefan_boltz

    solidus    = tctx.solidus
    liquidus   = tctx.liquidus
    latent     = tctx.latent

    base_power = tctx.base_power
    r_beam     = tctx.r_beam
    laser_loc  = tctx.laser_loc
    laser_on   = tctx.laser_on

    element_birth = tctx.element_birth
    node_birth    = tctx.node_birth
    surface_birth = tctx.surface_birth

    dt         = tctx.dt
    steps      = tctx.steps
    BOT_NODES  = tctx.BOT_NODES

    def thermal_step(carry, t_control):
        temperature, temperatures = carry
        t, control_t = t_control
        current_time = t * dt
        ae, an, asurf = update_birth(current_time, element_birth, node_birth, surface_birth)

        cps = jax.vmap(calc_cp, in_axes=(0, None, None, None, None, None, None))(
                    elements, temperature, Nip_ele, solidus, liquidus, cp_val, latent)
        m_vec = jnp.zeros(n_n)
        rhs = jnp.zeros(n_n)
        m_vec, rhs = update_mvec_stiffness(m_vec, rhs, cps, conds, elements, temperature, 
                                           ae, an, nodes, Bip_ele, Nip_ele, density)
        rhs = update_fluxes(t, rhs, temperature, base_power, control_t, asurf,
                            surfaces, surf_detJacs, surface_xy, surface_flux, Nip_sur,
                            nodes, laser_loc, laser_on[t], r_beam, h_conv, ambient, emissivity)

        update = dt * rhs / (m_vec + 1e-8) * an    
        temperature = temperature + update
        temperature = temperature.at[BOT_NODES].set(ambient)
        temperatures = temperatures.at[t].set(temperature)
        return (temperature, temperatures), None

    t_seq = jnp.arange(0, steps)
    control_seq = control
    carry = (
        jnp.full((n_n,), ambient),
        jnp.full((steps, n_n), ambient)
    )
    (temperature_final, temperatures), _ = jax.lax.scan(thermal_step, carry, (t_seq, control_seq))
    return temperatures