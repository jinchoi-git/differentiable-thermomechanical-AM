import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple
from .thermal import update_birth

class MechState(NamedTuple):
    U: jnp.ndarray
    E: jnp.ndarray
    Ep_prev: jnp.ndarray
    Hard_prev: jnp.ndarray
    dU: jnp.ndarray

@dataclass(frozen=True)
class MechContext:
    # sizes
    n_n: int
    n_e: int
    n_q: int

    # --- time / loop ---
    dt: float
    steps: int
    
    element_birth : jnp.ndarray
    node_birth    : jnp.ndarray
    surface_birth : jnp.ndarray

    # geometry / mesh
    elements: jnp.ndarray
    nodes: jnp.ndarray
    Nip_ele: jnp.ndarray
    Bip_ele: jnp.ndarray

    # constants / materials
    ambient: float
    poisson: float
    a1: float
    young1: jnp.ndarray
    temp_young1: jnp.ndarray
    Y1: jnp.ndarray
    temp_Y1: jnp.ndarray
    scl1: jnp.ndarray
    temp_scl1: jnp.ndarray
    BOT_HEIGHT: float
    Maxit: int

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
    #jax.debug.print("🔎 norm_SD = {ns}", ns=norm_SD)
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

# --- Implicit-Adjoint Newton Solve (custom VJP) --- #
# ---- core helper (NOT decorated) ----
def _newton_core(
    U_init, temperature_ip, E_th,
    elements, nodes,
    ele_K, ele_B, B_T, ele_D, ele_detJac,
    shear, bulk, a, Y,
    Ep_prev, Hard_prev,
    mask_e, Q_dof, elem_dofs,
    maxit, tikh=5e-6,
):
    def _assemble_global(K_tangent, elem_dofs, n_dof, mask_e):
        K_global = jnp.zeros((n_dof, n_dof))
        def body(K, e_idx):
            Ke = K_tangent[e_idx] * mask_e[e_idx]
            dofs = elem_dofs[e_idx]
            return K.at[dofs[:, None], dofs[None, :]].add(Ke), None
        K_global, _ = jax.lax.scan(body, K_global, jnp.arange(K_tangent.shape[0]))
        return K_global

    def _apply_dirichlet(K_global, R, Q_dof):
        K_global = K_global * (Q_dof[:, None] * Q_dof[None, :])
        K_global = K_global + jnp.diag(1.0 - Q_dof)
        R = R * Q_dof
        return K_global, R

    def _compute_K_R(U_it):
        # strains, constitutive, etc…
        E_base = jax.vmap(compute_E, in_axes=(0, 0, None))(elements, ele_B, U_it)
        E_corr = (E_base - E_th) * mask_e[:, None, None]
        S, DS, _, _, _ = constitutive_problem(E_corr, Ep_prev, Hard_prev, shear, bulk, a, Y)

        # K_tangent = ele_K + sum_q B^T[(detJ*DS) - ele_D]B
        D_diff    = (ele_detJac[:, :, None, None] * DS) - ele_D
        B_T_D_B   = jnp.sum(B_T @ D_diff @ ele_B, axis=1)
        K_tangent = ele_K + B_T_D_B

        # internal force residual R
        F_e = jnp.einsum("eqik,eqk->ei", B_T, ele_detJac[..., None] * S) * mask_e[:, None]
        n_dof = Q_dof.shape[0]
        R_node = jnp.zeros((n_dof,))
        R_node = R_node.at[elem_dofs.flatten()].add(F_e.flatten())

        # assemble the consistent tangent (unregularized), with BCs
        K_unreg = _assemble_global(K_tangent, elem_dofs, n_dof, mask_e)
        K_unreg, R = _apply_dirichlet(K_unreg, R_node, Q_dof)   # rows/cols masked + identity on fixed

        # add Tikhonov ONLY for the primal Newton step
        K_reg = K_unreg + tikh * jnp.eye(n_dof)

        return K_unreg, K_reg, R, E_corr, S

    def newton_iteration(i, U_it):
        K_unreg, K_reg, R, _, _ = _compute_K_R(U_it)
        dU = jnp.linalg.solve(K_reg, -R).reshape(U_it.shape)
        return U_it + dU

    U_star = jax.lax.fori_loop(0, maxit, newton_iteration, U_init)
    K_unreg_star, K_reg_star, R_star, E_corr_star, S_star = _compute_K_R(U_star)
    saved = (U_star, K_unreg_star, K_reg_star, R_star, E_corr_star, S_star,
             temperature_ip, E_th, elements, nodes,
             ele_K, ele_B, B_T, ele_D, ele_detJac,
             shear, bulk, a, Y,
             Ep_prev, Hard_prev, mask_e, Q_dof, elem_dofs, tikh)
    return U_star, saved
    
# ---- custom_vjp wrapper ----
@jax.custom_vjp
def newton_solve_implicit(
    U_init, temperature_ip, E_th,
    elements, nodes,
    ele_K, ele_B, B_T, ele_D, ele_detJac,
    shear, bulk, a, Y,
    Ep_prev, Hard_prev,
    mask_e, Q_dof, elem_dofs,
    maxit=2
):
    # IMPORTANT: return ONLY the primal!
    U_star, _ = _newton_core(
        U_init, temperature_ip, E_th,
        elements, nodes,
        ele_K, ele_B, B_T, ele_D, ele_detJac,
        shear, bulk, a, Y,
        Ep_prev, Hard_prev,
        mask_e, Q_dof, elem_dofs,
        maxit
    )
    return U_star

def newton_solve_implicit_fwd(
    U_init, temperature_ip, E_th,
    elements, nodes,
    ele_K, ele_B, B_T, ele_D, ele_detJac,
    shear, bulk, a, Y,
    Ep_prev, Hard_prev,
    mask_e, Q_dof, elem_dofs,
    maxit=2
):
    # Call the core helper (NOT the decorated function)
    U_star, saved = _newton_core(
        U_init, temperature_ip, E_th,
        elements, nodes,
        ele_K, ele_B, B_T, ele_D, ele_detJac,
        shear, bulk, a, Y,
        Ep_prev, Hard_prev,
        mask_e, Q_dof, elem_dofs,
        maxit
    )
    return U_star, saved

def newton_solve_implicit_bwd(saved, bar_Ustar):
    (U_star, K_unreg_star, K_reg_star, _R_star, _E_corr_star, _S_star,
     temperature_ip, E_th, elements, nodes,
     ele_K, ele_B, B_T, ele_D, ele_detJac,
     shear, bulk, a, Y,
     Ep_prev, Hard_prev, mask_e, Q_dof, elem_dofs, tikh) = saved

    # mask cotangent on Dirichlet dofs
    bar_u = bar_Ustar.reshape(-1) * Q_dof

    # 1) Adjoint solve with the **same** operator used in primal Newton
    lam = jnp.linalg.solve(K_reg_star.T, bar_u)
    # keep lam consistent with BCs (redundant but safe):
    lam = lam * Q_dof

    # 2) VJP of the masked residual wrt inputs, at fixed U*
    def R_masked(temperature_ip_, E_th_, Ep_prev_, Hard_prev_):
        # Strain at converged U*
        E_base = jax.vmap(compute_E, in_axes=(0, 0, None))(elements, ele_B, U_star)
        E_corr = (E_base - E_th_) * mask_e[:, None, None]

        # Stress only; residual needs S, not K
        S, _, _, _, _ = constitutive_problem(E_corr, Ep_prev_, Hard_prev_, shear, bulk, a, Y)

        # Internal force assembly (no K), then apply same BC mask as primal
        F_e    = jnp.einsum("eqik,eqk->ei", B_T, ele_detJac[..., None] * S) * mask_e[:, None]
        ndof   = Q_dof.shape[0]
        R_node = jnp.zeros((ndof,))
        R_node = R_node.at[elem_dofs.flatten()].add(F_e.flatten())
        return R_node * Q_dof 

    vjp_fn = jax.vjp(R_masked, temperature_ip, E_th, Ep_prev, Hard_prev)[1]
    g_temp_ip, g_Eth, g_Ep, g_Hard = vjp_fn(lam)
    g_U_init = jnp.zeros_like(U_star)
    Z = lambda x: jax.tree_util.tree_map(jnp.zeros_like, x)

    return (
        g_U_init,
        -g_temp_ip,
        -g_Eth,
        Z(elements), Z(nodes),
        Z(ele_K), Z(ele_B), Z(B_T), Z(ele_D), Z(ele_detJac),
        Z(shear), Z(bulk), Z(a), Z(Y),
        -g_Ep, -g_Hard,
        Z(mask_e), Z(Q_dof), Z(elem_dofs),
        None,  # maxit
    )

newton_solve_implicit.defvjp(newton_solve_implicit_fwd, newton_solve_implicit_bwd)

def mech(
    temperature,
    active_element_inds,
    active_node_inds,
    U,
    E,
    Ep_prev,
    Hard_prev,
    dU,
    current_time,
    mctx,
):
    
    # --- pull from context (no behavior change) --- #
    n_n = mctx.n_n
    n_e = mctx.n_e
    n_q = mctx.n_q
    elements = mctx.elements
    nodes = mctx.nodes
    Nip_ele = mctx.Nip_ele
    Bip_ele = mctx.Bip_ele
    ambient = mctx.ambient
    poisson = mctx.poisson
    a1 = mctx.a1
    young1 = mctx.young1
    temp_young1 = mctx.temp_young1
    Y1 = mctx.Y1
    temp_Y1 = mctx.temp_Y1
    scl1 = mctx.scl1
    temp_scl1 = mctx.temp_scl1
    BOT_HEIGHT = mctx.BOT_HEIGHT
    Maxit = mctx.Maxit

    # Masks
    mask_e = active_element_inds                # (n_e,)
    mask_n = active_node_inds                   # (n_n,)
    n_dof = n_n * 3

    # # Interpolate temperature at integration points
    temperature_ip = (
        Nip_ele[:, jnp.newaxis, :] @ temperature[elements][:, jnp.newaxis, :, jnp.newaxis].repeat(8, axis=1)
    )[:, :, 0, 0]
    # tau = 50.0  
    # temperature_ip = temperature_ip - tau * jax.nn.softplus((temperature_ip - 2300.0) / tau)

    # Material properties
    young = jnp.interp(temperature_ip, temp_young1, young1)
    shear = young / (2 * (1 + poisson))
    bulk = young / (3 * (1 - 2 * poisson))
    scl = jnp.interp(temperature_ip, temp_scl1, scl1)
    Y = jnp.interp(temperature_ip, temp_Y1, Y1)
    a = a1 * jnp.ones_like(young)

    # Thermal strain
    alpha_Th = jnp.zeros((n_e, n_q, 6)).at[:, :, 0:3].set(scl[:, :, None].repeat(3, axis=2))
    E_th = (temperature_ip[:, :, None].repeat(6, axis=2) - ambient) * alpha_Th
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
    Q_node = jnp.where(nodes[:, 2] < BOT_HEIGHT, 0.0, 1.0) * mask_n   # (n_n,)
    Q_dof  = jnp.repeat(Q_node, 3)                              # (n_dof,)
    elem_dofs = jnp.repeat(elements * 3, 3, axis=1) + jnp.tile(jnp.arange(3), (n_e, 8))

    # --- NEW: implicit Newton with custom VJP ---
    U_guess = U * mask_n[:, None]
    U_it = newton_solve_implicit(
        U_guess,                 # U_init
        temperature_ip,          # (n_e, n_q)
        E_th,                    # (n_e, n_q, 6)
        elements, nodes,
        ele_K, ele_B, B_T, ele_D, ele_detJac,
        shear, bulk, a, Y,
        Ep_prev, Hard_prev,
        mask_e, Q_dof, elem_dofs,
        maxit=Maxit
    )
           
    dU = U_it - U

    # Final stress for output
    E_base = jax.vmap(compute_E, in_axes=(0, 0, None))(elements, ele_B, U_it)
    E_corr = (E_base - E_th) * mask_e[:, None, None]
    S_final, DS, IND_p, Ep_new, Hard_new = constitutive_problem(E_corr, Ep_prev, Hard_prev, shear, bulk, a, Y)
    
    # Update global U
    U = jax.lax.dynamic_update_slice(U, U_it, (0, 0))

    return (
        S_final,
        U,
        E_corr,
        Ep_new,
        Hard_new,
        dU,
    )

# --- Mechanical simulation ---
def simulate_mechanics(temperatures, mctx):
    # ⬅️ pull sizes/time and schedules from context, not globals
    n_n = mctx.n_n
    n_e = mctx.n_e
    n_q = mctx.n_q
    dt  = mctx.dt
    steps = mctx.steps

    element_birth = mctx.element_birth
    node_birth    = mctx.node_birth
    surface_birth = mctx.surface_birth

    def mech_scan_step(state, t):
        T_t = temperatures[t]
        current_time = t * dt
        ae, an, _ = update_birth(current_time, element_birth, node_birth, surface_birth)

        S, U, E, Ep_new, Hard_new, dU = mech(
            T_t, ae, an, state.U, state.E, state.Ep_prev, state.Hard_prev,
            state.dU, current_time, mctx
        )
        new_state = MechState(U, E, Ep_new, Hard_new, dU)
        return new_state, S

    initial_mech_state = MechState(
        U=jnp.zeros((n_n, 3)),
        E=jnp.zeros((n_e, n_q, 6)),
        Ep_prev=jnp.zeros((n_e, n_q, 6)),
        Hard_prev=jnp.zeros((n_e, n_q, 6)),
        dU=jnp.zeros((n_n, 3)),
    )

    # your downsampled mechanics steps:
    mech_timesteps = jnp.arange(0, steps, 10)

    final_state, S_seq = jax.lax.scan(mech_scan_step, initial_mech_state, mech_timesteps)
    return S_seq

def simulate_mechanics_forward(temperatures, mctx):
    # ⬅️ pull sizes/time and schedules from context, not globals
    n_n = mctx.n_n
    n_e = mctx.n_e
    n_q = mctx.n_q
    dt  = mctx.dt
    steps = mctx.steps

    element_birth = mctx.element_birth
    node_birth    = mctx.node_birth
    surface_birth = mctx.surface_birth

    def mech_scan_step(state, t):
        T_t = temperatures[t]
        current_time = t * dt
        ae, an, _ = update_birth(current_time, element_birth, node_birth, surface_birth)

        S, U, E, Ep_new, Hard_new, dU = mech(
            T_t, ae, an, state.U, state.E, state.Ep_prev, state.Hard_prev,
            state.dU, current_time, mctx
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

    # your downsampled mechanics steps:
    mech_timesteps = jnp.arange(0, steps, 10)

    final_state, (S_seq, U_seq) = jax.lax.scan(mech_scan_step, initial_mech_state, mech_timesteps)
    return S_seq, U_seq
