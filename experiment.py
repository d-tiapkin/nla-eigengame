import functools
import sys
import optax
import numpy as np

import jax
import jax.numpy as jnp
from jaxline import base_config
from jaxline import experiment
from jaxline import platform
from jaxline import utils as jl_utils

from typing import Dict, List, Mapping, Optional

import algorithms.eigengame_mu as mu

class EigengameExperiment(experiment.AbstractExperiment):
    def init(self, mode, init_rng, config):
        """Initialization function for a Jaxline experiment."""
        super(EigengameExperiment, self).__init__(mode=mode, init_rng=init_rng)

        weights = np.eye(self._total_k) * 2 - np.ones((self._total_k, self._total_k))
        weights[jnp.triu_indices(self._total_k, 1)] = 0
        self._weights = jnp.reshape(weights, [self._num_devices,
                                            self._k_per_device, 
                                            self._total_k])
                                            
        local_rng = jax.random.fold_in(jax.random.PRNGkey(172), jax.host_id())
        keys = jax.random.split(local_rng, self._num_devices)
        V = jax.pmap(lambda key: jax.random.normal(key, (self._k_per_device, self._dims)))(keys)
        self._V = jax.pmap(lambda V: V / jnp.linalg.norm(V, axis=1, keepdims=True))(V)    
            
        # Define parallel update function. If k_per_device is not None, wrap individual functions with vmap here.
        self._partial_grad_update = functools.partial(
            self._grads_and_update, axis_groups=self._axis_index_groups)
        self._par_grad_update = jax.pmap(
            self._partial_grad_update, in_axes=(0, 0, None, 0, 0, 0), axis_name='i')
            
        self._optimizer = optax.sgd(learning_rate=1e-4, momentum=0.9, nesterov=True)

    def step(self, *, global_step, rng, writer):
        """Step function for a Jaxline experiment"""
        key = jax.random.PRNGKey(17)
        X = jax.random.normal(key, shape=(50,50))
        inputs = X
        self._local_V = jnp.reshape(self._V, (self._total_k, self._dims))  
        self._V, self._opt_state, utilities, lr = self._par_grad_update(
            self._V, self._weights, self._local_V, inputs, self._opt_state,
            global_step)

    def evaluate(self, *, global_step: jnp.ndarray, rng: jnp.ndarray, writer: Optional[jl_utils.Writer]) -> Optional[Dict[str, jnp.ndarray]]:
        return super().evaluate(global_step, rng, writer)

    def _grads_and_update(self, vi, weights, eigs, input, opt_state, axis_index_groups):
        """Compute utilities and update directions, psum and apply.
        Args:
        vi: shape (d,), eigenvector to be updated
        weights:  shape (k_per_device, k,), mask for penalty coefficients,
        eigs: shape (k, d), i.e., vectors on rows
        input: shape (N, d), minibatch X_t
        opt_state: optax state
        axis_index_groups: For multi-host parallelism https://jax.readthedocs.io/en/latest/_modules/jax/_src/lax/parallel.html 
        Returns:
        vi_new: shape (d,), eigenvector to be updated
        opt_state: new optax state
        utilities: shape (1,), utilities
        """
        grads, utilities = self._grads_and_utils(vi, weights, V, input)
        avg_grads = jax.lax.psum(
            grads, axis_name='i', axis_index_groups=axis_index_groups)
        vi_new, opt_state, lr = self._update_with_grads(vi, avg_grads, opt_state)
        return vi_new, opt_state, utilities

    def _grads_and_utils(self, vi, weights, V, inputs):
        """Compute utiltiies and update directions ("grads"). 
            Wrap in jax.vmap for k_per_device dimension."""
        utilities = mu.utility(vi, weights, V, inputs)
        grads = mu.eg_grads(vi, weights, V, inputs)
        return grads, utilities
        
    def _update_with_grads(self, vi, grads, opt_state):
        """Compute and apply updates with optax optimizer.
            Wrap in jax.vmap for k_per_device dimension."""
        updates, opt_state = self._optimizer.update(-grads, opt_state)
        vi_new = optax.apply_updates(vi, updates)
        vi_new /= jnp.linalg.norm(vi_new)
        return vi_new, opt_state

if __name__ == '__main__':
    platform.main(EigengameExperiment, sys.argv[1:])