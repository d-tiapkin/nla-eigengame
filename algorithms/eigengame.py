import numpy as np
import optax

import jax
import jax.numpy as jnp


def utility(vi, weights, eigs, data):
    """Compute Eigengame utilities.
        util: shape (1,), utility for vi
    """
    data_vi = jnp.dot(data, vi)
    data_eigs = jnp.transpose(jnp.dot(data, jnp.transpose(eigs)))  # Xvj on row j
    vi_m_vj2 = jnp.dot(data_eigs, data_vi)**2.
    vj_m_vj = jnp.sum(data_eigs * data_eigs, axis=1)
    r_ij = vi_m_vj2 / vj_m_vj
    util = jnp.dot(jnp.array(r_ij), weights)
    return util


def eg_grads_mu(vi: jnp.ndarray,
                weights: jnp.ndarray,
                eigs: jnp.ndarray,
                data: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
     vi: shape (d,), eigenvector to be updated
     weights:  shape (k,), mask for penalty coefficients,
     eigs: shape (k, d), i.e., vectors on rows
     data: shape (N, d), minibatch X_t
    Returns:
     grads: shape (d,), gradient for vi
    """
    weights_ij = (jnp.sign(weights + 0.5) - 1.) / 2.  # maps -1 to -1 else to 0
    data_vi = jnp.dot(data, vi)
    data_eigs = jnp.transpose(jnp.dot(data,
                            jnp.transpose(eigs)))  # Xvj on row j
    vi_m_vj = jnp.dot(data_eigs, data_vi)
    penalty_grads = vi_m_vj * jnp.transpose(eigs)
    penalty_grads = jnp.dot(penalty_grads, weights_ij)
    grads = jnp.dot(jnp.transpose(data), data_vi) + penalty_grads
    return grads


def eg_grads_alpha(vi: jnp.ndarray,
                weights: jnp.ndarray,
                eigs: jnp.ndarray,
                data: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
     vi: shape (d,), eigenvector to be updated
     weights:  shape (k,), mask for penalty coefficients,
     eigs: shape (k, d), i.e., vectors on rows
     data: shape (N, d), minibatch X_t
    Returns:
     grads: shape (d,), gradient for vi
    """
    weights_ij = (jnp.sign(weights + 0.5) - 1.) / 2.  # maps -1 to -1 else to 0
    data_vi = jnp.dot(data, vi)
    data_eigs = jnp.transpose(jnp.dot(data,
                            jnp.transpose(eigs)))  # Xvj on row j
    vi_m_vj = jnp.dot(data_eigs, data_vi)
    vj_m_vj = jnp.diag(jnp.dot(data_eigs, data_eigs.T))
    penalty_grads = vi_m_vj / vj_m_vj * data_eigs.T
    penalty_grads = jnp.dot(penalty_grads, weights_ij)
    grads = jnp.dot(jnp.transpose(data), data_vi + penalty_grads)
    return grads


def compute_eigengame(data, total_k, type="mu", lr=1e-4, beta=0.0, n_epoch=100, batch_size=32, seed=172, callback=None):
    """
        Compute top-k right singular vector of matrix data, using Eigengame algorihtm
        Args:
            data: shape (M, d)
            total_k: int, should be multiplier of number of LAX-devices
            type: str, type of algorithm, changes the way to compute gradients
            lr: float, learning rate for SGD optimizer
            beta: float, momentum coefficient for SGD optimizer
            n_epoch: int, number of iterations 
            batch_size: int
            seed: int
            callback: callable(V, utilities), generic callback function
        Returns:
            V - top-k right singular vectors
    """
    if type == "mu":
        compute_grads = eg_grads_mu
    elif type == "alpha":
        compute_grads = eg_grads_alpha
    else:
        raise ValueError("Unknown type of EigenGame algorithm")

    num_devices = jax.local_device_count()
    k_per_device = int(np.ceil(total_k // num_devices))
    dim = data.shape[1]
    m = data.shape[0]
    block_size = batch_size * num_devices
    if block_size <= m:
        # we can do it in one pass
        batch_size = -1


    local_rng = jax.random.PRNGKey(seed)
    keys = jax.random.split(local_rng, num_devices)
    V = jax.pmap(lambda key: jax.random.normal(key, (k_per_device, dim)))(keys)
    optimizer = optax.sgd(learning_rate=lr, momentum=beta, nesterov=True)
    opt_state = optimizer.init(V)

    weights = jnp.eye(total_k) * 2 - jnp.ones((total_k, total_k))
    weights = weights.at[jnp.triu_indices(total_k, 1)].set(0).reshape([num_devices, k_per_device, total_k])

    def update(vis, weights, eigs, data, opt_state):
        """Compute utilities and update directions and apply.
            Args:
                vis: shape (k_per_device, d,), eigenvectors to be updated
                weights:  shape (k_per_device, k,), mask for penalty coefficients,
                eigs: shape (k, d), i.e., vectors on rows
                data: shape (N, d), minibatch X_t
                opt_state: optax state
            Returns:   block_size = batch_size * num_devices
or to be updated
                opt_state: new optax state
                utilities: shape (1,), utilities
        """
        data = data.reshape(-1, dim)
        grads = jnp.zeros_like(vis)
        grads = jax.lax.fori_loop(0, k_per_device,\
            lambda i, val: val.at[i].set(compute_grads(vis[i], weights[i], eigs, data)), init_val=grads)
        utilities = jax.lax.fori_loop(0, k_per_device,\
            lambda i, val: val + utility(vis[i], weights[i], eigs, data), init_val=0)

        updates, opt_state = optimizer.update(-grads, opt_state, vis)
        vi_new = optax.apply_updates(vis, updates)
        vi_new /= jnp.linalg.norm(vi_new)
        
        return vi_new, opt_state, utilities
    
    if batch_size == -1:
        in_axes = (0,0,None,None,0)
    else:
        in_axes = (0,0,None,0,0)
    parallel_update = jax.pmap(update, in_axes=in_axes, axis_name='i')

    for epoch in range(n_epoch):
        if batch_size == -1:
            local_V = jnp.reshape(V, (total_k, dim)).copy()
            V, opt_state, utilities = parallel_update(V, weights, local_V, data, opt_state)
            continue


        key, local_rng = jax.random.split(local_rng)
    
        last_block_start = (m // block_size) * block_size
        new_data = jax.random.permutation(key, data, axis=0)[:last_block_start,:]
        new_data = new_data.reshape([m // block_size, num_devices, batch_size, dim])
        
        for it in range(m // block_size):
            cur_data = new_data[it]
            local_V = jnp.reshape(V, (total_k, dim)).copy()
            V, opt_state, utilities = parallel_update(V, weights, local_V, cur_data, opt_state)

        if callback is not None:
            callback(epoch, local_V, utilities)

    return V.reshape(total_k, dim)