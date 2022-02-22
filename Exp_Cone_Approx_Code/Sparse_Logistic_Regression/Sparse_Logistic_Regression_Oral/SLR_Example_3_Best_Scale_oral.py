# Problem: Sparse Logistic Regression
# Dataset: Oral
# Method: Example 3 Best Scale

from gurobipy import *
import numpy as np 
import sys, itertools
import orthopy
from numpy import linalg as LA
import math
import time
import pandas as pd
global df
df = pd.DataFrame(columns=('pp','n', 'd','kbar','kk', 'a','scale','UB','LB', 'Time', 'GAP', 'Evaulated Obj', 'Approx_Gap'))

def logistic(X,y,kbar,lamb,kk,a,sa):
    n, d = int(X.shape[0]), int(X.shape[1])
    M = Model()
    bigM=10.0
    u = M.addMVar(n,  lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    v = M.addMVar(n,  lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    f = M.addMVar(n,  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    t = M.addMVar(n,  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    theta  = M.addMVar(d, lb=-10.0, ub=10.0, vtype=GRB.CONTINUOUS)
    mu  = M.addMVar(d, lb=0, ub=10.0, vtype=GRB.CONTINUOUS)
    zvar  = M.addMVar(d, lb=0.0, ub=1, vtype=GRB.BINARY)
    vu = M.addMVar((n,kk),  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    ru1 = M.addMVar((n,kk),  lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    ru2 = M.addMVar((n,kk),  lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    vu3 = M.addMVar((n,kk),  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    vv = M.addMVar((n,kk),  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    rv1 = M.addMVar((n,kk),  lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    rv2 = M.addMVar((n,kk),  lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    vv3 = M.addMVar((n,kk),  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    points, weights = orthopy.line.schemes.legendre(kk, decimal_places=30)  
    points=points
    weights=[float(weights[k]) for k in range(kk)]
    Weights=np.matrix(weights)
    points=[float(points[k])+1.0 for k in range(kk)]
    MP=np.diag(points)
    spoints=[math.pow(2.0**(a)*(float(points[k])),0.5) for k in range(kk)]
    MSP=np.diag(spoints)
    M.setObjective((float(1.0/n))*t.sum()+lamb*mu.sum(), GRB.MINIMIZE)
    M.addConstr(zvar.sum()==kbar)
    M.addConstr(theta<=np.diag([bigM]*d)@zvar)
    M.addConstr(theta>=-np.diag([bigM]*d)@zvar)
    M.addConstr(mu<=np.diag([bigM]*d)@zvar)
    M.addConstr(mu>=theta)
    M.addConstr(mu>=-theta)
    M.addConstr((np.array(np.diag(1.0-2.0*y)*np.matrix(X)))@theta==f)
    M.addConstr(u+v<=np.array([1]*n))
    M.addConstrs((Weights @ vu[i,:])+sa*math.log(2)>=-t[i] for i in range(n))
    M.addConstrs((Weights @ vv[i,:])+sa*math.log(2)>=f[i]-t[i] for i in range(n))
    M.addConstrs((a*(points[k]**(a-1.0))*(u[i]/(2**sa)-1)-(2.0**a)*vu[i,k] ==ru2[i,k] for k in range(kk) for i in range(n)))
    M.addConstrs((a*(points[k]**(a-1.0))*(v[i]/(2**sa)-1)-(2.0**a)*vv[i,k] ==rv2[i,k] for k in range(kk) for i in range(n)))
    M.addConstrs((a-points[k]*vu[i,k] ==ru1[i,k] for k in range(kk) for i in range(n)))
    M.addConstrs((a-points[k]*vv[i,k] ==rv1[i,k] for k in range(kk) for i in range(n)))
    M.addConstrs((MSP[k,k] *vu[i,k]==vu3[i,k] for k in range(kk)for i in range(n)))
    M.addConstrs((MSP[k,k] *vv[i,k]==vv3[i,k] for k in range(kk)for i in range(n)))
    M.addConstrs((ru1[i,k]@ru2[i,k]>= vu3[i,k]@vu3[i,k] for k in range(kk)for i in range(n)))
    M.addConstrs((rv1[i,k]@rv2[i,k]>= vv3[i,k]@vv3[i,k] for k in range(kk)for i in range(n)))
    return M, zvar,theta

def logistic2(X,y,kbar,lamb,kk,a,sa1,sa2,obj1):
    n, d = int(X.shape[0]), int(X.shape[1])
    M = Model()
    bigM=min(n*np.log(2)/lamb,obj1/lamb)
    u = M.addMVar(n,  lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    v = M.addMVar(n,  lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    f = M.addMVar(n,  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    t = M.addMVar(n,  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    theta  = M.addMVar(d, lb=-bigM, ub=bigM, vtype=GRB.CONTINUOUS)
    mu  = M.addMVar(d, lb=0, ub=bigM, vtype=GRB.CONTINUOUS)
    zvar  = M.addMVar(d, lb=0.0, ub=1, vtype=GRB.BINARY)
    vu = M.addMVar((n,kk),  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    ru1 = M.addMVar((n,kk),  lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    ru2 = M.addMVar((n,kk),  lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    vu3 = M.addMVar((n,kk),  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    vv = M.addMVar((n,kk),  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    rv1 = M.addMVar((n,kk),  lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    rv2 = M.addMVar((n,kk),  lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    vv3 = M.addMVar((n,kk),  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    points, weights = orthopy.line.schemes.legendre(kk, decimal_places=30)  
    points=points
    weights=[float(weights[k]) for k in range(kk)]
    Weights=np.matrix(weights)
    points=[float(points[k])+1.0 for k in range(kk)]
    MP=np.diag(points)
    spoints=[math.pow(2.0**(a)*(float(points[k])),0.5) for k in range(kk)]
    MSP=np.diag(spoints)
    M.setObjective((float(1.0/n))*t.sum()+lamb*mu.sum(), GRB.MINIMIZE)
    M.addConstr(zvar.sum()==kbar)
    M.addConstr(theta<=np.diag([bigM]*d)@zvar)
    M.addConstr(theta>=-np.diag([bigM]*d)@zvar)
    M.addConstr(mu<=np.diag([bigM]*d)@zvar)
    M.addConstr(mu>=theta)
    M.addConstr(mu>=-theta)
    M.addConstr((np.array(np.diag(1.0-2.0*y)*np.matrix(X)))@theta==f)
    M.addConstr(u+v<=np.array([1]*n))
    M.addConstrs((Weights @ vu[i,:])+float(sa1[0,i])>=-t[i] for i in range(n))
    M.addConstrs((Weights @ vv[i,:])+float(sa2[0,i])>=f[i]-t[i] for i in range(n))
    M.addConstrs((a*(points[k]**(a-1.0))*(u[i]/(np.exp(sa1[0,i]))-1)-(2.0**a)*vu[i,k] ==ru2[i,k] for k in range(kk) for i in range(n)))
    M.addConstrs((a*(points[k]**(a-1.0))*(v[i]/(np.exp(sa2[0,i]))-1)-(2.0**a)*vv[i,k] ==rv2[i,k] for k in range(kk) for i in range(n)))
    M.addConstrs((a-points[k]*vu[i,k] ==ru1[i,k] for k in range(kk) for i in range(n)))
    M.addConstrs((a-points[k]*vv[i,k] ==rv1[i,k] for k in range(kk) for i in range(n)))
    M.addConstrs((MSP[k,k] *vu[i,k]==vu3[i,k] for k in range(kk)for i in range(n)))
    M.addConstrs((MSP[k,k] *vv[i,k]==vv3[i,k] for k in range(kk)for i in range(n)))
    M.addConstrs((ru1[i,k]@ru2[i,k]>= vu3[i,k]@vu3[i,k] for k in range(kk)for i in range(n)))
    M.addConstrs((rv1[i,k]@rv2[i,k]>= vv3[i,k]@vv3[i,k] for k in range(kk)for i in range(n)))
    return M, zvar,theta

def loaddata(filename,pp):
    x, y = [], []
    Xy=pd.read_excel(filename+'.xlsx',sheet_name=filename, index_col=None)  
    y=Xy[Xy.columns[0]].values.tolist()
    x=Xy[Xy.columns[1:]].values.tolist()
    y=y[0:pp]
    x=x[0:pp]
    return np.array(x), np.array(y)
    
def Evaluation(X,y,kbar,lamb,zvar,theta):
    n, d = int(X.shape[0]), int(X.shape[1])
    z=np.array([float(zvar[j].x) for j in range(d)])
    theta=np.array([float(theta[j].x) for j in range(d)])
    supp=[i for i in range(d) if z[i]>0.5]
    obj=float(lamb*sum([abs(theta[i]) for i in supp]))
    theta=np.matrix(theta[supp]).transpose()
    XX=np.matrix(X[:,supp])
    t=np.diag(1.0-2.0*y)*XX*theta
    t=np.log(1.0+np.exp(t))
    obj+=float(sum(sum(t))/n)
    return obj

def Evaluation2(X,y,kbar,lamb,zvar,theta):
    n, d = int(X.shape[0]), int(X.shape[1])
    z=np.array([float(zvar[j].x) for j in range(d)])
    theta=np.array([float(theta[j].x) for j in range(d)])
    supp=[i for i in range(d) if z[i]>0.5]
    obj=float(lamb*sum([abs(theta[i]) for i in supp]))
    theta=np.matrix(theta[supp]).transpose()
    XX=np.matrix(X[:,supp])
    f=np.diag(1.0-2.0*y)*XX*theta
    t=np.log(1.0+np.exp(f))
    return (-t).transpose(),(f-t).transpose(),obj+float(sum(sum(t))/n)

np.random.seed(0)
instance_number=0
a=1.0
for pp in range(100, 1100, 100): 
    x, y = loaddata('oral_toxicity',pp) 
    X=x
    n, d = int(X.shape[0]), int(X.shape[1])
    kbar=20
    lamb=0.01
    for sa,kk in [(x,y) for x in [-2.0] for y in [1]]:
        M, zvar, theta = logistic(X,y,kbar,lamb,kk,a,sa)
        M.params.timelimit = 30
        start0 = time.time()
        M.optimize()
        Mtime0 = time.time() - start0
        sa1,sa2,obj1 = Evaluation2(X,y,kbar,lamb,zvar,theta)
        M, zvar, theta = logistic2(X,y,kbar,lamb,kk,a,sa1,sa2,obj1)
        M.params.timelimit = 570
        start = time.time()
        M.optimize()
        Mtime = time.time() - start
        UB = M.objVal
        LB = M.objBound
        Gap = 1.0-LB/(UB+1e-7)
        Mtime_total = Mtime + Mtime0
        EUB = Evaluation(X,y,kbar,lamb,zvar,theta)
        Approx_Gap = abs(UB-EUB)/EUB
        df.loc[instance_number] = np.array([pp,n,d,kbar,kk,a,sa, UB, LB, Mtime_total, Gap, EUB, Approx_Gap])
        df.to_csv('exmp3_best_scale_sparse_logistic_regression_oral.csv')
        instance_number = instance_number + 1
            
        try:
            print("The true obj is %f" %(Evaluation(X,y,kbar,lamb,zvar,theta)))
        except:
            print("Could not get solution")


