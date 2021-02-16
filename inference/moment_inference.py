# moment-based inference
def MomentInference(vals, export_moments=False):
    # code from Anton Larsson's R implementation
    from scipy import stats # needs imports inside function when run in ipyparallel
    import numpy as np
    m1 = float(np.mean(vals))
    m2 = float(sum(vals*(vals - 1))/len(vals))
    m3 = float(sum(vals*(vals - 1)*(vals - 2))/len(vals))
    
    # sanity check on input (e.g. need at least on expression level)
    if sum(vals) == 0: return np.nan
    if m1 == 0: return np.nan
    if m2 == 0: return np.nan
    
    r1=m1
    r2=m2/m1
    r3=m3/m2
    
    if (r1*r2-2*r1*r3 + r2*r3) == 0: return np.nan
    if ((r1*r2 - 2*r1*r3 + r2*r3)*(r1-2*r2+r3)) == 0: return np.nan
    if (r1 - 2*r2 + r3) == 0: return np.nan
        
    lambda_est = (2*r1*(r3-r2))/(r1*r2-2*r1*r3 + r2*r3)
    mu_est = (2*(r3-r2)*(r1-r3)*(r2-r1))/((r1*r2 - 2*r1*r3 + r2*r3)*(r1-2*r2+r3))
    v_est = (2*r1*r3 - r1*r2 - r2*r3)/(r1 - 2*r2 + r3)
    
    if export_moments:
        return np.array([lambda_est, mu_est, v_est, r1, r2, r3])
    
    return np.array([lambda_est, mu_est, v_est])