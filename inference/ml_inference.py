def MaximumLikelihood(vals, export_asymp_ci = False, fix = 0, export_fun = False):
    from inference.moment_inference import MomentInference
    from scipy.interpolate import interp1d
    from scipy.optimize import minimize
    from scipy import special
    from scipy.stats import poisson,norm
    from scipy.special import j_roots
    from scipy.special import beta as beta_fun    
    import numpy as np
    if len(vals) == 0:
        return np.array([np.nan, np.nan, np.nan])
    def dBP(at, alpha, bet, lam):
        at.shape = (len(at), 1)
        np.repeat(at, 50, axis = 1)
        def fun(at, m):
            if(max(m) < 1e6):
                return(poisson.pmf(at,m))
            else:
                return(norm.pdf(at,loc=m,scale=sqrt(m)))
        
        x,w = j_roots(50,alpha = bet - 1, beta = alpha - 1)
        gs = np.sum(w*fun(at, m = lam*(1+x)/2), axis=1)
        prob = 1/beta_fun(alpha, bet)*2**(-alpha-bet+1)*gs
        return(prob)
    def LogLikelihood(x, vals):
        kon = x[0]
        koff = x[1]
        ksyn = x[2]
        return(-np.sum(np.log( dBP(vals,kon,koff,ksyn) + 1e-10) ) )
    x0 = MomentInference(vals)
    if np.isnan(x0).any():
        x0 = np.array([10,10,10])
    bnds = ((1e-3,1e3),(1e-3,1e3), (1, 1e10))
    vals_ = np.copy(vals) # Otherwise the structure is violated.
    try:
        ll = minimize(LogLikelihood, x0, args = (vals_), method='L-BFGS-B', bounds = bnds)
    except:
        if export_fun:
            return np.array([np.nan,np.nan,np.nan]), np.nan
        return np.array([np.nan,np.nan,np.nan])
    #se = ll.hess_inv.todense().diagonal()
    if export_fun:
        return ll.x, ll.fun
    estim = ll.x
    return estim