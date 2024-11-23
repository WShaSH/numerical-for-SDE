import numpy as np
def a2(t,y):
    a = 1
    return a * y

def b2(t,y):
    b = 1
    return b

def dat2(t,y):
    a = 1
    return 0

def dbt2(t,y):
    b = 1
    return 0

def dax2(t,y):
    a = 1
    return a

def dbx2(t,y):
    b = 1
    return 0

def ddax2(t,y):
    a = 1
    return 0

def ddbx2(t,y):
    b = 1
    return 0

def true_solver2(t,T,x0,dWt):
    #r = -1
    a = 1
    b = 1
    X_T = x0*np.exp(a*T)+b*np.dot(dWt,np.exp(a*(T-t.reshape(-1,1)))[0:-1])
    return X_T.ravel()