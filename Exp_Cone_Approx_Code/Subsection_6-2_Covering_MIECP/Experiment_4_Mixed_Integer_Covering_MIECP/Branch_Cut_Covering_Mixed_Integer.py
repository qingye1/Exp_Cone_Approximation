# Problem: Covering
# Problem type: Mixed Integer
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
        M.addConstr(A@x>=b)
        M.addConstrs(x[j]-xsol[j]==t[j] for j in range(n))
        M.params.OutputFlag = 0
        M.optimize()
        return np.matrix([float(x[j].x) for j in range(n)]).transpose()
    def Grad(x):
        x=x.reshape((n, 1))
        t1=c*x
        t=1+np.log(t1)
        grad=np.diagflat(t)*c
        return grad,np.diagflat(t1)*np.log(t1)
    def lazycuts(M,where):
        if where == GRB.callback.MIPSOL:
            xsol = M.cbGetSolution(x)
            xsol=np.matrix([xsol[j]for j in range(n)]).transpose()
            grad, f0=Grad(xsol)
            for i in range(p):
                M.cbLazy(f0[i,0]+quicksum(grad[i,j]*(x[j]-xsol[j]) for j in range(n))<=t[i])
    M = Model()
    t = M.addMVar(p,  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    x  = M.addMVar(n, lb=0.0, ub=1, vtype=GRB.CONTINUOUS)
    M.setObjective(t.sum(), GRB.MINIMIZE)
    M.addConstr(A@x>=b)
    xsol=[1]*n
    xsol=np.matrix(xsol).transpose()
    grad, f0=Grad(xsol)
    const=np.array((grad*xsol).transpose())[0]
    grad=np.array(grad)
    f0=np.array(f0.transpose())[0]
    M.addConstr(f0+grad @x-const<=t)
    M.params.timelimit = 3600 
    start=time.time()
    maxiter=100
    LB=0
    UB=1e30
    Gap=1.0
    iteration=0
    step=1.0
    M.params.OutputFlag = 0
    x[int(n/2):n].vtype = GRB.BINARY
    check=0
    while (iteration<=maxiter) and (UB-LB)/abs(UB)>1e-4 and time.time() - start<=3600:
        UB=min(UB,float(sum(f0)))
        print("at iteration %d, the current upper bound is %f, current lower bound is %f" %(iteration, UB,LB))
        M.optimize()
        xsol=np.matrix([float(x[j].x) for j in range(n)]).transpose()
        xsol=xsol.reshape((n, 1))
        LB=M.objval
        grad, f0=Grad(xsol)
        const=np.array((grad*xsol).transpose())[0]
        grad=np.array(grad)
        f0=np.array(f0.transpose())[0]
        M.addConstr(f0+grad @x-const<=t)
        iteration+=1
    print("at iteration %d, the current upper bound is %f, current lower bound is %f" %(iteration, UB,LB))
    RTime=time.time() - start
    MTime = time.time() - start
    return M, x,RTime, MTime,UB,LB

def Evaluation(c,x,n,p):
    x=[float(x[j].x) for j in range(n)]
    t=c*np.matrix(x).transpose()
    t0=np.log(t)
    obj=float(t.transpose()*t0)
    return obj

np.random.seed(0)
m=100
n=200
kk=3
instance_number=0
for p in range(10, 110, 10):
    c=np.random.randint(10, size=(p, n))/float(n)
    A=np.random.randint(10, size=(m, n))
    b=np.array([2*n]*m)
    M, x, Rtime, Mtime , UB, LB= knap(c,A,b,m,n,p)
    Gap=1.0-LB/(UB+1e-7)
    EUB=Evaluation(c,x,n,p)
    Approx_Gap=abs(UB-EUB)/EUB
    df.loc[instance_number] =np.array([m,n,p, UB, LB, Rtime, Mtime, Gap, EUB, Approx_Gap])                 
    df.to_csv('gurobi_covering_branch_cut_mixed.csv')
    instance_number=instance_number+1

    try:
        print([float(x[i].x) for i in range(n)])
        print("The true obj is %f" %(Evaluation(c,x,n,p)))
    except:
        print("Could not get solution")
      
