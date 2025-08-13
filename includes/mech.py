import jax
import jax.numpy as jnp
import numpy as np

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