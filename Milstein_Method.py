import numpy as np
import matplotlib as plt
def Milstein(t0,T,y0,ndt,M,f,g,dg,dWt):
    #np.random.seed(1)
    dt = ( T - t0 ) / ndt
    t = np.linspace(t0, T, ndt+1)
    y = np.zeros((M,ndt+1))
    y[:,0] = y0
    for i in range(1,ndt+1):
        y[:,i] = y[:,i-1] + f(t[i-1],y[:,i-1]) * dt + \
                g(t[i-1],y[:,i-1]) * dWt[:,i-1] + \
                 1/2 * g(t[i-1],y[:,i-1]) * dg(t[i-1],y[:,i-1]) * (dWt[:,i-1]**2-dt)
    return y