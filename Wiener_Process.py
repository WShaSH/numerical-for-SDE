import numpy as np

def Wiener_Process(t0,T,ndt,M):
    np.random.seed(1)
    dt = ( T - t0 ) / ndt
    t = np.linspace(t0 , T , ndt + 1)
    Nor = np.random.randn( M , ndt )
    #np.random.seed(2)
    #Nor2 = np.random.randn(M, ndt)
    dWt = np.sqrt(dt) * Nor
    Wt = np.cumsum( dWt , axis = 1 )
    Wt = np.hstack( (np.zeros((M,1)),Wt))
    #Nort = np.cumsum( Nor2 , axis = 1 )
    #np.hstack((np.zeros((M, 1)), Nort))
    return t , Wt , dWt, Nor#, Nort