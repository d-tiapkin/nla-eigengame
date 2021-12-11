"""
Copyright 2020 DeepMind Technologies Limited. 


Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import jax
import optax
import jax.numpy as jnp

def eg_grads(vi: jnp.ndarray,
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

