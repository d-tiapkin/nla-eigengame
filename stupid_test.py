import jax
import jax.numpy as jnp
import logging

import algorithms.eigengame as eg
import algorithms.utils as utils
import time
import matplotlib.pyplot as plt

import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

key = jax.random.PRNGKey(17)
X = jnp.diag(jnp.arange(1000))

V = eg.compute_eigengame(X, 4, type='mu', batch_size=32, lr=1e-4, beta=0.0, n_epoch=1000)
U, Sigma = utils._SVD_U_Sigma(X, V)
X_approx1 = (U * Sigma) @ V

U, Sigma, Vt =  jnp.linalg.svd(X)
Sigma = Sigma.at[4:].set(0)
X_approx2 = (U * Sigma) @ Vt

fig = plt.figure()
print(jnp.linalg.norm(X_approx1 - X_approx2) / jnp.linalg.norm(X_approx2))
plt.spy(X_approx1 - X_approx2, precision=1e-5)
plt.show()
plt.savefig('eig.pdf', dpi=150)
