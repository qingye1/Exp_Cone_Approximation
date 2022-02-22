# Problem: Packing
# Problem type: Mixed Integer
# Method: Example 3

from gurobipy import *
import numpy as np 
import sys, itertools
import orthopy
from numpy import linalg as LA
import math
import time
import pandas as pd
global df
df = pd.DataFrame(columns=('m', 'n','p','kk', 'a','scale','UB', 'LB', 'Time', 'Gap', 'Evaulated Obj', 'Approx_Gap'))

def knap(c,A,b,m,n,p,kk,a,sa):
    M = Model()
    t = M.addMVar(p,  lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    x = M.addMVar(n, lb=0.0, ub=1, vtype=GRB.CONTINUOUS)
    x[int(n/2):n].vtype = GRB.BINARY
    v = M.addMVar((p,kk),  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    r1 = M.addMVar((p,kk),  lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    r2 = M.addMVar((p,kk),  lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    v3 = M.addMVar((p,kk),  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    points, weights = orthopy.line.schemes.legendre(kk, decimal_places=30)  
    points=points
    weights=[float(weights[k]) for k in range(kk)]
    Weights=np.matrix(weights)
    points=[float(points[k])+1.0 for k in range(kk)]
    MP=np.diag(points)
    spoints=[math.pow(2.0**(a)*(float(points[k])),0.5) for k in range(kk)]
    MSP=np.diag(spoints)
    M.setObjective(t.sum(), GRB.MINIMIZE)
    M.addConstr(A@x<=b)
    M.addConstrs((Weights @ v[i,:])+sa*math.log(2)>=(c[i,:] @ x) for i in range(p))
    M.addConstrs((a*(points[k]**(a-1.0))*(t[i]/(2**sa)-1)-(2.0**a)*v[i,k] ==r2[i,k] for k in range(kk) for i in range(p)))
    M.addConstrs((a-points[k]*v[i,k] ==r1[i,k] for k in range(kk) for i in range(p)))
    M.addConstrs((MSP[k,k] *v[i,k]==v3[i,k] for k in range(kk)for i in range(p)))
    M.addConstrs((r1[i,k]@r2[i,k]>= v3[i,k]@v3[i,k] for k in range(kk)for i in range(p)))
    return M, x

def Evaluation(c,x,n,p):
    x=[float(x[j].x) for j in range(n)]
    t=c*np.matrix(x).transpose()
    t=np.exp(t)
    obj=float(sum(t))
    return obj

np.random.seed(0)
m=100
n=200
kk=3
a=1.0
instance_number=0
for p in range(10, 110, 10):
    c=-np.random.randint(10, size=(p, n))/float(n)
    A=np.random.randint(10, size=(m, n))
    b=np.array([2*n]*m)
    for sa,kk in [(x,y) for x in [-4] for y in [3]]:
        M, x = knap(c,A,b,m,n,p,kk,a,sa)
        M.params.timelimit = 3600
        start=time.time()
        M.optimize()
        Mtime = time.time() - start
        UB=M.objVal
        LB=M.objBound
        Gap=1.0-LB/(UB+1e-7)
        EUB=Evaluation(c,x,n,p)
        Approx_Gap=abs(UB-EUB)/EUB
        df.loc[instance_number] =np.array([m,n,p,kk,a,sa, UB, LB, Mtime, Gap, EUB, Approx_Gap])
        df.to_csv('gurobi_packing_exmp3_mixed.csv')
        instance_number=instance_number+1
            
        try:
            print([float(x[i].x) for i in range(n)])
            print("The true obj is %f" %(Evaluation(c,x,n,p)))
            
        except:
            print("Could not get solution")
            
