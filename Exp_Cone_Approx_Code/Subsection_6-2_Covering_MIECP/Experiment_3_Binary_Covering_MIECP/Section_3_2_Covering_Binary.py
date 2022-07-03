# Problem: Covering
# Problem type: Binary
# Method: Section 3.2

from gurobipy import *
import numpy as np 
import sys, itertools
import orthopy
from numpy import linalg as LA
import math
import time
import pandas as pd
global df
df = pd.DataFrame(columns=('m', 'n','p','N', 'UB','LB', 'Time', 'GAP', 'Evaulated Obj', 'Approx_Gap'))

def knap(c,A,b,m,n,p,N):
    M = Model()
    t = M.addMVar(p,  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    ct = M.addMVar(p,  lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    dt = M.addMVar(p,  lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    x  = M.addMVar(n, lb=0.0, ub=1, vtype=GRB.BINARY)
    r = M.addMVar((p,N),  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    M.setObjective(t.sum(), GRB.MINIMIZE)
    M.addConstr(A@x>=b)
    M.addConstr(c@x==ct)
    M.addConstrs(r[i,N-1]*2**N==((2**N)*ct[i]-t[i]+1.0/(2**(N+1))*dt[i]) for i in range(p))
    M.addConstrs(dt[i]@ct[i]>= (t[i])@(t[i])for i in range(p))
    M.addConstrs((ct[i]>=r[i,0]@r[i,0] for i in range(p)))
    M.addConstrs((r[i,k]>=0 for k in range(N-1) for i in range(p)))
    M.addConstrs((r[i,k] @ ct[i]>= r[i,k+1] @ r[i,k+1] for k in range(N-1) for i in range(p)))
    return M, x

def Evaluation(c,x,n,p):
    x=[float(x[j].x) for j in range(n)]
    t=c*np.matrix(x).transpose()
    t0=np.log(t)
    obj=float(t.transpose()*t0)
    return obj

np.random.seed(0)
m=100
n=50
N=3
instance_number=0
for p in range(5, 55, 5):
    c=np.random.randint(10, size=(p, n))/float(n)
    A=np.random.randint(10, size=(m, n))
    b=np.array([2*n]*m)
    for N in [6]:
        M, x = knap(c,A,b,m,n,p,N)        
        M.params.timelimit = 3600 
        start=time.time()
        M.optimize()
        UB=M.objVal
        LB=M.objBound
        Gap=1.0-LB/UB
        Mtime = time.time() - start
        EUB=Evaluation(c,x,n,p)       
        Approx_Gap=abs(UB-EUB)/EUB       
        df.loc[instance_number] =np.array([m,n,p,N, UB, LB, Mtime, Gap, EUB, Approx_Gap])
        df.to_csv('gurobi_covering_section_3_2_binary.csv')
        instance_number=instance_number+1
            
        try:
            print([float(x[i].x) for i in range(n)])
            print("The true obj is %f" %(Evaluation(c,x,n,p)))            
        except:
            print("Could not get solution")
            sys.exit(1)
