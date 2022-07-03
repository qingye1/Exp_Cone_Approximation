# Problem: Sparse Logistic Regression
# Dataset: Oral
# Method: Mosek

from mosek.fusion import *
import numpy as np 
import sys, itertools
import time
import pandas as pd
global df
df = pd.DataFrame(columns=('pp','n', 'd','kbar','UB','LB', 'Time', 'GAP', 'Evaulated Obj','Approx_Gap'))

def softplus(M, t, u):
    n = t.getShape()[0]
    z1 = M.variable(n)
    z2 = M.variable(n)
    M.constraint(Expr.add(z1, z2), Domain.equalsTo(1))
    M.constraint(Expr.hstack(z1, Expr.constTerm(n, 1.0), Expr.sub(u,t)), Domain.inPExpCone())
    M.constraint(Expr.hstack(z2, Expr.constTerm(n, 1.0), Expr.neg(t)), Domain.inPExpCone())

def logisticRegression(X, y, kbar,lamb):
    n, d = int(X.shape[0]), int(X.shape[1])
    bigM = n*np.log(2)/lamb
    kbar = min(kbar,d)
    M = Model()
    theta = M.variable(d)
    t = M.variable(n)
    reg = M.variable(d, Domain.greaterThan(0))
    zvar = M.variable(d, Domain.binary())
    M.objective(ObjectiveSense.Minimize, Expr.add(Expr.mul(float(1/n),Expr.sum(t)),
                                                  Expr.mul(lamb,Expr.sum(reg))))
    M.constraint(Expr.sum(zvar), Domain.equalsTo(kbar))
    M.constraint(Expr.sub(theta,Expr.mul(np.diag([bigM]*d),zvar)), Domain.lessThan(0))
    M.constraint(Expr.sub(reg,Expr.mul(np.diag([bigM]*d),zvar)), Domain.lessThan(0))
    M.constraint(Expr.add(theta,Expr.mul(np.diag([bigM]*d),zvar)), Domain.greaterThan(0))
    M.constraint(Expr.add(reg,theta), Domain.greaterThan(0))
    M.constraint(Expr.sub(reg,theta), Domain.greaterThan(0))
    signs = list(map(lambda y: -1.0 if y==1 else 1.0, y))
    softplus(M, t, Expr.mulElm(Expr.mul(X, theta), signs))
    return M, theta,zvar

def readdata(filename):
    Xy=pd.read_csv(filename+'.csv',index_col=None)  
    x_data=Xy[Xy.columns[1:]].values.tolist()
    y_data=Xy[Xy.columns[0]].values.tolist()
    return x_data, y_data
    
def loaddata(x_data, y_data, pp):
    x=x_data[0:pp]
    y=y_data[0:pp]
    return np.array(x), np.array(y)
    
def Evaluation(X,y,kbar,lamb,zvar,theta):
    n, d = int(X.shape[0]), int(X.shape[1])
    z=zvar.level()
    theta=theta.level()
    supp=[i for i in range(d) if z[i]>0.5]
    obj=float(lamb*sum([abs(theta[i]) for i in supp]))
    theta=np.matrix(theta[supp]).transpose()
    XX=np.matrix(X[:,supp])
    t=np.diag(1.0-2.0*y)*XX*theta
    t=[np.log(1.0+np.exp(t))]
    obj+=float(sum(sum(t))/n)
    return obj

np.random.seed(0)
instance_number=0
filename = 'oral_toxicity'
x_data, y_data = readdata(filename)
for pp in range(100, 1100, 100): 
    x, y = loaddata(x_data, y_data, pp)
    X = x
    n, d = int(X.shape[0]), int(X.shape[1])
    kbar=20
    lamb=0.01
    M, theta,zvar = logisticRegression(X, y, kbar,lamb)
    M.setLogHandler(sys.stdout)
    M.setSolverParam('mioMaxTime', 600.0)
    start=time.time()
    M.solve()
    Mtime=time.time()-start
    M.acceptedSolutionStatus(AccSolutionStatus.Feasible)
    Gap=M.getSolverDoubleInfo("mioObjRelGap")
    LB=M.getSolverDoubleInfo("mioObjBound")
    UB=LB+M.getSolverDoubleInfo("mioObjAbsGap")
    
    try:
        print(theta.level())
        print(zvar.level())
        EUB=Evaluation(X,y,kbar,lamb,zvar,theta)
        print(EUB)
        Approx_Gap=abs(UB-EUB)/EUB
        df.loc[instance_number] =np.array([pp,n,d,kbar, UB, LB, Mtime, Gap, EUB,Approx_Gap])
        df.to_csv('mosek_sparse_logistic_regression_oral.csv')
        instance_number=instance_number+1

    except:
        print("Could not get solution")
        EUB=UB
        Approx_Gap=abs(UB-EUB)/EUB
        df.loc[instance_number] =np.array([pp,n,d,kbar, UB, LB, Mtime, Gap, EUB,Approx_Gap])
        df.to_csv('mosek_sparse_logistic_regression_oral.csv')
        instance_number=instance_number+1
