# Problem: Packing
# Problem type: Continuous
# Method: Mosek

from mosek.fusion import *
import numpy as np 
import sys, itertools
import orthopy
from numpy import linalg as LA
import math
import time
import pandas as pd
from output_file_names_packing_continuous_small_scale import output_name_Mosek
global df
df = pd.DataFrame(columns=('m', 'n','p','UB','LB', 'Time', 'GAP', 'Evaulated Obj','Approx_Gap'))

def knap(c,A,b,m,n,p):
    M = Model()
    t = M.variable(p, Domain.greaterThan(0))
    x  = M.variable(n,Domain.greaterThan(0))   
    cc=Matrix.dense(c)
    AA=Matrix.dense(A)
    bb=Matrix.dense([b])
    M.objective(ObjectiveSense.Minimize,Expr.sum(t))
    M.constraint(Expr.sub(Expr.mul(AA,x),bb.transpose()), Domain.lessThan(0))
    M.constraint(Expr.hstack(t, Expr.constTerm(p, 1.0), Expr.mul(cc,x)), Domain.inPExpCone())
    M.constraint(x, Domain.lessThan(1))
    return M, x

def Evaluation(c,x,n,p):
    x=x.level()
    t=c*np.matrix(x).transpose()
    t=np.exp(t)
    obj=float(sum(t))
    return obj

np.random.seed(0)
m=100
instance_number=0
for n in range(100, 1100, 100):
    for p in range(10, 110, 10):
        c=-np.random.randint(10, size=(p, n))/float(n)
        A=np.random.randint(10, size=(m, n))
        b=np.array([4*n]*m)            
        M, x = knap(c,A,b,m,n,p)
        M.setLogHandler(sys.stdout)
        M.setSolverParam('mioMaxTime', 3600.0)
        start=time.time()
        M.solve()
        Mtime=time.time()-start
        Gap=M.getSolverDoubleInfo("mioObjRelGap")
        LB=M.dualObjValue()
        UB=M.primalObjValue()
    
        try:
            print(x.level())               
            EUB=Evaluation(c,x,n,p)                    
            Approx_Gap=abs(UB-EUB)/EUB            
            df.loc[instance_number] =np.array([m,n,p, UB, LB, Mtime, Gap, EUB,Approx_Gap])
            df.to_csv(output_name_Mosek+'.csv')
            instance_number=instance_number+1                
        except:
            print("Could not get solution")
            EUB=UB
            Approx_Gap=abs(UB-EUB)/EUB            
            df.loc[instance_number] =np.array([m,n,p, UB, LB, Mtime, Gap, EUB,Approx_Gap])
            df.to_csv(output_name_Mosek+'.csv')
            instance_number=instance_number+1
