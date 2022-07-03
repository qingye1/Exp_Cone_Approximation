# Problem: Packing
# Problem type: Binary
# Method: Section 3.1 Shift

from gurobipy import *
import numpy as np 
import sys, itertools
import orthopy
from numpy import linalg as LA
import math
import time
import pandas as pd
from output_file_names_packing_binary_small_scale import output_name_Sec31_Shift
global df
df = pd.DataFrame(columns=('m', 'n','p','N', 'UB','LB', 'Time','total Time', 'GAP', 'Evaulated Obj', 'Approx_Gap'))

def knap(c,A,b,m,n,p,N,t0):
    M = Model()
    t = M.addMVar(p,  lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    x  = M.addMVar(n, lb=0.0, ub=1, vtype=GRB.BINARY)
    r = M.addMVar((p,N),  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    t0=np.array(t0)[:,0]
    t1=np.exp(t0)
    print(t1)
    M.setObjective(t1@t, GRB.MINIMIZE)
    M.addConstr(A@x<=b)
    M.addConstrs(r[i,N-1]*2**N==(2**N+c[i,:] @ x-t0[i]) for i in range(p))
    M.addConstrs((t[i]>=r[i,0]@r[i,0] for i in range(p)))
    M.addConstrs((r[i,k]>= r[i,k+1]@r[i,k+1] for k in range(N-1) for i in range(p)))
    return M, x

def knap0(c,A,b,m,n,p,kk,a,sa):
    M = Model()
    t = M.addMVar(p,  lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    x  = M.addMVar(n, lb=0.0, ub=1, vtype=GRB.BINARY)
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

def Evaluation2(c,x,n,p):
    x=[float(x[j].x) for j in range(n)]
    t=c*np.matrix(x).transpose()
    return t

np.random.seed(0)
m=100
n=20
N=3
instance_number=0
for p in range(5, 27, 2):
    c=-np.random.randint(10, size=(p, n))/float(n)
    A=np.random.randint(10, size=(m, n))
    b=np.array([4*n]*m)
    M, x = knap0(c,A,b,m,n,p,1,1,-4)
    M.params.timelimit = 2
    start0=time.time()
    M.optimize()
    Mtime0 = time.time() - start0
    t0=Evaluation2(c,x,n,p)
    for N in [1, 3, 5]:
        M, x = knap(c,A,b,m,n,p,N,t0)
        M.params.timelimit = 3600 
        start=time.time()
        M.optimize()
        UB=M.objVal
        LB=M.objBound
        Gap=1.0-LB/UB
        Mtime = time.time() - start
        Mtime_total=Mtime0+Mtime
        EUB=Evaluation(c,x,n,p)
        Approx_Gap=abs(UB-EUB)/EUB
        df.loc[instance_number] =np.array([m,n,p,N, UB, LB, Mtime, Mtime_total,Gap, EUB, Approx_Gap])
        df.to_csv(output_name_Sec31_Shift+'.csv')
        instance_number=instance_number+1
            
        try:
            print([float(x[i].x) for i in range(n)])
            print("The true obj is %f" %(Evaluation(c,x,n,p)))          
        except:
            print("Could not get solution")

