import jax
import jax.numpy as jnp
import logging

import algorithms.eigengame_alpha as alpha
import algorithms.utils as utils
import time

key = jax.random.PRNGKey(17)
X = jax.random.normal(key, shape=(1000,1000))

start = time.time()
U, Sigma, V = alpha.calc_eigengame_eigenvectors(X, 3, num_iter=1000)
finish = time.time()
our_time = finish-start

start = time.time()
U, Sigma, Vt =  jnp.linalg.svd(X)
finish = time.time()
svd_time = finish-start

print('Our algorithm: {}, SVD from numpy: {}'.format(our_time, svd_time))

