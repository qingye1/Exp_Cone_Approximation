# Problem: Covering
# Problem type: Mixed Integer
# Method: Section 3.2 Shift

from gurobipy import *
import numpy as np 
import sys, itertools
import orthopy
from numpy import linalg as LA
import math
import time
import pandas as pd
global df
df = pd.DataFrame(columns=('m', 'n','p','N', 'UB', 'LB', 'Time', 'GAP', 'Evaulated Obj', 'Approx_Gap'))

def knap(c,A,b,m,n,p,N,t0):
    M = Model()
    t = M.addMVar(p,  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    ct = M.addMVar(p,  lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    dt = M.addMVar(p,  lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    et = M.addMVar(p,  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    x  = M.addMVar(n, lb=0.0, ub=1, vtype=GRB.CONTINUOUS)
    x[int(n/2):n].vtype = GRB.BINARY
    M.setObjective(t.sum(), GRB.MINIMIZE)
    M.addConstr(A@x>=b)
    M.addConstr(c@x==ct)
    if N>=1:
        r = M.addMVar((p,N),  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        M.addConstrs(r[i,N-1]*2**N==((2**N+t0[i])*ct[i]-t[i]+1.0/(2**(N+1))*dt[i]) for i in range(p))
        M.addConstrs(dt[i]@ct[i]>= et[i]@et[i] for i in range(p))
        M.addConstrs((t[i]-t0[i]*ct[i]==et[i])for i in range(p))
        M.addConstrs((ct[i]*np.exp(t0[i])>=r[i,0]@r[i,0] for i in range(p)))
        M.addConstrs((r[i,k]>=0 for k in range(N-1) for i in range(p)))
        M.addConstrs((r[i,k] @ ct[i]>= r[i,k+1] @ r[i,k+1] for k in range(N-1) for i in range(p)))
    else:
        M.addConstrs(dt[i]@ct[i]>= et[i]@et[i] for i in range(p))
        M.addConstrs((t[i]-t0[i]*ct[i]==et[i])for i in range(p))
        M.addConstrs((np.exp(t0[i])>=((1+t0[i])*ct[i]-t[i]+1.0/(2**(0+1))*dt[i])  for i in range(p)))
    return M, x

def knap0(c,A,b,m,n,p,kk,a,sa):
    M = Model()
    t = M.addMVar(p,  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    x  = M.addMVar(n, lb=0.0, ub=1, vtype=GRB.CONTINUOUS)
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
    M.addConstr(A@x>=b)
    M.addConstrs((Weights @ v[i,:])+sa*math.log(2)*(c[i,:] @ x)>=-t[i] for i in range(p))
    M.addConstrs((a*(points[k]**(a-1.0))*(1.0/(2**sa)-(c[i,:] @ x))-(2.0**a)*v[i,k] ==r2[i,k] for k in range(kk) for i in range(p)))
    M.addConstrs((a*(c[i,:] @ x)-points[k]*v[i,k] ==r1[i,k] for k in range(kk) for i in range(p)))
    M.addConstrs((MSP[k,k] *v[i,k]==v3[i,k] for k in range(kk)for i in range(p)))
    M.addConstrs((r1[i,k]@r2[i,k]>= v3[i,k]@v3[i,k] for k in range(kk)for i in range(p)))
    return M, x

def Evaluation(c,x,n,p):
    x=[float(x[j].x) for j in range(n)]
    t=c*np.matrix(x).transpose()
    t0=np.log(t)
    obj=float(t.transpose()*t0)
    return obj

def Evaluation2(c,x,n,p):
    x=[float(x[j].x) for j in range(n)]
    t=c*np.matrix(x).transpose()
    t0=np.log(t)
    return t0

np.random.seed(0)
m=100
n=200
N=3
instance_number=0
for p in range(10, 110, 10):
    c=np.random.randint(10, size=(p, n))/float(n)
    A=np.random.randint(10, size=(m, n))
    b=np.array([2*n]*m)
    M, x = knap0(c,A,b,m,n,p,1,1,-1)
    M.params.timelimit = 2
    start0=time.time()
    M.optimize()
    Mtime0 = time.time() - start0
    t0=Evaluation2(c,x,n,p)
    for N in [0]:        
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
        df.loc[instance_number] =np.array([m,n,p,N, UB, LB, Mtime_total,Gap,  EUB, Approx_Gap])               
        df.to_csv('gurobi_covering_section_3_2_shift_mixed.csv')
        instance_number=instance_number+1      
        
        try:
            print([float(x[i].x) for i in range(n)])
            print("The true obj is %f" %(Evaluation(c,x,n,p)))           
        except:
            print("Could not get solution")
            sys.exit(1)
