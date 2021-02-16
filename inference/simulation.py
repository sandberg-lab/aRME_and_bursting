from numpy.random import beta, poisson
def sim_beta_poisson(kon,koff,ksyn=100, size=100):
    "Generate simulated single-cell RNA-seq data according to the two-state model and with parameters kon, koff, ksyn"
    v = beta(kon, koff, size)
    return poisson(v*ksyn)

def simBetaPoisson(params, size=100):
    kon = params[0]
    koff = params[1]
    ksyn = params[2]
    v = beta(kon, koff, size)
    return poisson(v*ksyn)