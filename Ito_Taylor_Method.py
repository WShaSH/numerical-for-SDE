import numpy as np
import matplotlib as plt
def Ito_Taylor15(t0,T,y0,ndt,M,a,b,dat,dbt,dax,dbx,ddax,ddbx,dWt,Nor):
    #np.random.seed(2)
    dt = ( T - t0 ) / ndt
    dZ = 1/2 * dt**(3/2) * ( Nor + (1/np.sqrt(3)) * np.random.randn(np.shape(Nor)[0],np.shape(Nor)[1]) )
    t = np.linspace(t0, T, ndt+1)
    y = np.zeros((M,ndt+1))
    y[:,0] = y0
    #print('a')
    for i in range(1,ndt+1):
        y[:,i] = y[:,i-1] + a(t[i-1],y[:,i-1]) * dt + \
                b(t[i-1],y[:,i-1]) * dWt[:,i-1] + \
                 1/2 * b(t[i-1],y[:,i-1]) * dbx(t[i-1],y[:,i-1]) * (dWt[:,i-1]**2-dt) + \
            dt**2/2 * ( dat(t[i-1],y[:,i-1]) + a(t[i-1],y[:,i-1]) * dax(t[i-1],y[:,i-1]) + \
                        1/2 * b(t[i-1],y[:,i-1])**2 * ddax(t[i-1],y[:,i-1]) ) + b(t[i-1],y[:,i-1]) * dax(t[i-1],y[:,i-1]) * dZ[:,i-1]\
                 + ( dbt(t[i-1],y[:,i-1]) + a(t[i-1],y[:,i-1]) * dbx(t[i-1],y[:,i-1]) + 1/2 * b(t[i-1],y[:,i-1])**2 * ddbx(t[i-1],y[:,i-1]) ) \
                 * ( dt * dWt[:,i-1] - dZ[:,i-1] ) + 1/2 * ( b(t[i-1],y[:,i-1]) * dbx(t[i-1],y[:,i-1])**2 + b(t[i-1],y[:,i-1])**2 * ddbx(t[i-1],y[:,i-1])) * \
                  (1/3 * dWt[:,i-1]**2 - dt) * dWt[:,i-1]
        #print(y[:,i])
    return y