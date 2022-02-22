# Problem: Covering
# Problem type: Binary
# Method: Mosek

from mosek.fusion import *
import numpy as np 
import sys, itertools
import orthopy
from numpy import linalg as LA
import math
import time
import pandas as pd
global df
df = pd.DataFrame(columns=('m', 'n','p','UB','LB', 'Time', 'GAP', 'Evaulated Obj','Approx_Gap'))

def knap(c,A,b,m,n,p):
    M = Model()
    t = M.variable(p, Domain.unbounded())
    x  = M.variable(n,Domain.binary())
    cc=Matrix.dense(c)
    AA=Matrix.dense(A)
    bb=Matrix.dense([b])
    M.objective(ObjectiveSense.Minimize,Expr.sum(t))
    M.constraint(Expr.sub(Expr.mul(AA,x),bb.transpose()), Domain.greaterThan(0))
    M.constraint(Expr.hstack(Expr.constTerm(p, 1.0), Expr.mul(cc,x),Expr.sub(0,t)), Domain.inPExpCone())
    return M, x

def Evaluation(c,x,n,p):
    x=x.level()
    t=c*np.matrix(x).transpose()
    t0=np.log(t)
    obj=float(t.transpose()*t0)
    return obj

np.random.seed(0)
m=100
n=50
p=10
instance_number=0
for p in range(5, 55, 5):
    c=np.random.randint(10, size=(p, n))/float(n)
    A=np.random.randint(10, size=(m, n))
    b=np.array([2*n]*m)
    M, x = knap(c,A,b,m,n,p)
    M.setLogHandler(sys.stdout)
    M.setSolverParam('mioMaxTime', 3600.0)
    start=time.time()
    M.solve()
    Mtime=time.time()-start
    Gap=M.getSolverDoubleInfo("mioObjRelGap")
    LB=M.getSolverDoubleInfo("mioObjBound")
    UB=LB+M.getSolverDoubleInfo("mioObjAbsGap")
    
    try:
        print(x.level())
        EUB=Evaluation(c,x,n,p)
        Approx_Gap=abs(UB-EUB)/EUB
        df.loc[instance_number] =np.array([m,n,p, UB, LB, Mtime, Gap, EUB,Approx_Gap])
        df.to_csv('mosek_covering_binary.csv')
        instance_number=instance_number+1
    except:
        print("Could not get solution")
        EUB=UB
        Approx_Gap=abs(UB-EUB)/abs(EUB)
        df.loc[instance_number] =np.array([m,n,p, UB, LB, Mtime, Gap, EUB,Approx_Gap])
        df.to_csv('mosek_covering_binary_no_solution.csv')
        instance_number=instance_number+1
