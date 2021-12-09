import numpy as np
import jax
import jax.numpy as jnp
import logging

from algorithms.utils import SVDize


def sphere_grad(v,X,V1, k):
    rewards = jnp.dot(X, v)
    penalties = jnp.zeros(v.shape[0])
    for j in range(k):
        Xvj = jnp.dot(X,V1[:,j])
        c = jnp.dot(rewards,Xvj)/jnp.dot(Xvj, Xvj)
        penalties = penalties + c * Xvj
    return 2 * jnp.dot(jnp.transpose(X), rewards - penalties)

def model(v,X,V1, k):
    Xv = jnp.dot(X,v)
    rewards = jnp.dot(jnp.transpose(Xv), Xv)
    penalties = 0
    for j in range(k):
        Xvj = jnp.dot(X,V1[:,j].reshape(-1,1))
        penalties = penalties + (jnp.dot(jnp.transpose(Xv),Xvj))**2/jnp.dot(jnp.transpose(Xvj), Xvj)
    return jnp.sum(rewards-penalties)


def update(v,X,V1,k, lr=0.1,riemannian_projection=False):
    dv = sphere_grad(v,X,V1, k)
    if riemannian_projection:
        dvr = dv - (jnp.dot(dv.T,v))*v
        vhat = v+lr*dvr
    else:
        vhat = v+lr*dv
    return (vhat/jnp.linalg.norm(vhat))

@SVDize
def calc_eigengame_eigenvectors(X, K, num_iter=100):
    n = X.shape[0]
    v = jnp.ones(n)
    v = v/jnp.linalg.norm(v)
    v0 = jnp.ones(n)
    v0 = v0/jnp.linalg.norm(v0)
    V1 = jnp.zeros((n,K))
    V1 = V1.at[:,0].set(v)

    for k in range(K):
        logging.info("Finding the eigenvector {}".format(k))
        for i in range(num_iter):
            if k==0:
                v = update(v,X,V1,k)
            else:
                v = update(v,X,V1,k,riemannian_projection=False)
        V1 = V1.at[:,k].set(v)
        v = v0
    return V1
