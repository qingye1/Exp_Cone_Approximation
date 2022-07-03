# Problem: Packing
# Problem type: Binary
# Method: Branch and Cut

from gurobipy import *
import numpy as np 
import sys, itertools
import orthopy
from numpy import linalg as LA
import math
import time
import pandas as pd
global df
df = pd.DataFrame(columns=('m', 'n','p','UB','LB', 'RelaxTime','Time', 'GAP', 'Evaulated Obj', 'Approx_Gap'))

def knap(c,A,b,m,n,p):
    def Proj(xsol):
        M = Model()
        xsol=np.matrix(xsol)
        x  = M.addMVar(n, lb=0.0, ub=1, vtype=GRB.CONTINUOUS)
        t  = M.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        M.setObjective(t@t, GRB.MINIMIZE)
        M.addConstr(A@x<=b)
        M.addConstrs(x[j]-xsol[j]==t[j] for j in range(n))
        M.params.OutputFlag = 0
        M.optimize()
        return np.matrix([float(x[j].x) for j in range(n)]).transpose()
    def Grad(x):
        x=x.reshape((n, 1))
        t1=c*x
        t=np.exp(t1)
        grad=np.diagflat(t)*c
        return grad,t
    def lazycuts(M,where):
        if where == GRB.callback.MIPSOL:
            xsol = M.cbGetSolution(x)
            xsol=np.matrix([xsol[j]for j in range(n)]).transpose()
            grad, f0=Grad(xsol)
            for i in range(p):
                M.cbLazy(f0[i,0]+quicksum(grad[i,j]*(x[j]-xsol[j]) for j in range(n))<=t[i])
    M = Model()
    t = M.addVars(p,  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    x  = M.addVars(n, lb=0.0, ub=1, vtype=GRB.CONTINUOUS)
    M.setObjective(t.sum(), GRB.MINIMIZE)
    M.addConstrs(quicksum(A[i,j]*x[j] for j in range(n))<=b[i] for i in range(m))
    xsol=[0]*n
    xsol=np.matrix(xsol).transpose()
    grad, f0=Grad(xsol)
    const=np.array((grad*xsol).transpose())[0]
    for i in range(p):
        M.addConstr(f0[i,0]+quicksum(grad[i,j]*(x[j]-xsol[j]) for j in range(n))<=t[i])
    M.params.timelimit = 3600 
    start=time.time()
    maxiter=0
    LB=0
    UB=1e30
    Gap=1.0
    iteration=0
    step=1.0
    M.params.OutputFlag = 0
    check=0
    while (iteration<=maxiter) and (UB-LB)/abs(UB)>1e-4 and time.time() - start<=3600:
        UB=min(UB,float(sum(f0)))
        print("at iteration %d, the current upper bound is %f, current lower bound is %f" %(iteration, UB,LB))
        M.optimize()
        xsol=np.matrix([float(x[j].x) for j in range(n)]).transpose()
        xsol=xsol.reshape((n, 1))
        LB=M.objval
        grad, f0=Grad(xsol)
        for i in range(p):
            M.addConstr(f0[i,0]+quicksum(grad[i,j]*(x[j]-xsol[j]) for j in range(n))<=t[i])
        iteration+=1
    print("at iteration %d, the current upper bound is %f, current lower bound is %f" %(iteration, UB,LB))
  
    RTime=time.time() - start
    for j in range(n):
        x[j].vtype=GRB.BINARY 
    M.params.LazyConstraints = 1
    M.params.OutputFlag = 1
    M.optimize(lazycuts)
    MTime = time.time() - start
    return M, x,RTime, MTime

def Evaluation(c,x,n,p):
    x=[float(x[j].x) for j in range(n)]
    t=c*np.matrix(x).transpose()
    t=np.exp(t)
    obj=float(sum(t))
    return obj

np.random.seed(0)
m=100
n=100
kk=3

instance_number=0
for p in range(5, 55, 5):
    c=-np.random.randint(10, size=(p, n))/float(n)
    A=np.random.randint(10, size=(m, n))
    b=np.array([4*n]*m)
    M, x, Rtime, Mtime = knap(c,A,b,m,n,p)
    UB=M.objVal
    LB=M.objBound
    Gap=1.0-LB/(UB+1e-7)
    EUB=Evaluation(c,x,n,p)
    Approx_Gap=abs(UB-EUB)/EUB
    df.loc[instance_number] =np.array([m,n,p, UB, LB, Rtime, Mtime, Gap, EUB, Approx_Gap])
    df.to_csv('gurobi_packing_branch_cut_binary.csv')
    instance_number=instance_number+1
    
    try:
        print([float(x[i].x) for i in range(n)])
        print("The true obj is %f" %(Evaluation(c,x,n,p)))
        
    except:
        print("Could not get solution")
    
