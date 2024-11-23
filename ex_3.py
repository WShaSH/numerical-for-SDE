import numpy as np
def a(t,x):
    return -1 / (1+t) * x

def b(t,x):
    return 1 / (1+t)

def dat(t,x):
    return 1 / (1+t)**2 * x

def dbt(t,x):
    return -1 / (1+t)**2

def dax(t,x):
    return -1/(1+t)

def dbx(t,x):
    return 0

def ddax(t,x):
    return 0

def ddbx(t,x):
    return 0

def true_solver3(t,Wt):
    return Wt / (1+t)