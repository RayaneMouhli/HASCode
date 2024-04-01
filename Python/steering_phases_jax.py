from jax import numpy as jnp
import math

def steering_phases_PA8(v, h, z, R, ElemLoc, f, c0):
    thvect,phivect=ElemLoc[:,0],ElemLoc[:,1]
    numelem = thvect.shape[0]
    ang = jnp.zeros(numelem)
    kk = 2*jnp.pi*f/c0
    for g in range(numelem):
        tt = R * math.sin(thvect[g])
        
        ss2 = R * math.cos(thvect[g])
        ss3 = math.sin(phivect[g])
        
        s = ss2*ss3
        
        aa3 = math.cos(phivect[g])
        a = ss2 * aa3
        
        b = R - a
        d = R- (-z)
        
        r = math.sqrt((tt-v)**2 + (s+h)**2 + (d-b)**2)
        
        # ang[g]=kk*np.round(r-R)
        ang = ang.at[g].set(kk*round(r-R))
    return ang