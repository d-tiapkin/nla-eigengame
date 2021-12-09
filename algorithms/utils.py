import jax.numpy as jnp

def _SVD_U_Sigma(X, V):
    '''
        params:
            X - data matrix
            V - approximation of V-matrix from SVD
        output: matrix U and sigular values Sigma
    '''
    M = jnp.dot(X, V)
    Sigma = jnp.linalg.norm(M, axis=0)
    return M / Sigma, Sigma

def SVDize(func):
    '''
        Decorator that transform function that returns only V factor to SVD functions
        params:
            func - any function, that inputs matrix of size (n,m) and k (rank of approximation) and return V-factor from SVD
        output: SVD function
    '''
    def SVD(X, k, **kwargs):
        V = func(X, k, **kwargs)
        U, Sigma = _SVD_U_Sigma(X,V)
        return U, Sigma, V
    return SVD