import numpy as np
#
def a(t,y):
    mu = 1
    return mu * y

def b(t,y):
    sigma = 1
    return sigma * y

def dat(t,y):
    mu = 1
    return 0

def dbt(t,y):
    sigma = 1
    return 0

def dax(t,y):
    mu = 1
    return mu

def dbx(t,y):
    sigma = 1
    return sigma

def ddax(t,y):
    mu = 1
    return 0

def ddbx(t,y):
    sigma = 1
    return 0

def true_solver1(t,Wt):
    #r = -1
    mu = 1
    sigma = 1
    return np.exp((mu-1/2*sigma**2)*t+sigma*Wt)