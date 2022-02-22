# Problem: Sparse Logistic Regression
# Dataset: Student
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
df = pd.DataFrame(columns=('n', 'd','kbar','UB','LB', 'RelaxTime','Time', 'Total Time', 'GAP', 'Evaulated Obj', 'Approx_Gap'))

def logistic0(X,y,kbar,lamb,kk,a,sa):
    n, d = int(X.shape[0]), int(X.shape[1])
    M = Model()
    bigM=n*np.log(2)/lamb
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

def logistic(X,y,kbar,lamb,obj1):
    n, d = int(X.shape[0]), int(X.shape[1])    
    def Grad(z,tt):
        z=z.reshape((d, 1))
        tt=tt.reshape((d, 1))
        supp=[i for i in range(d) if z[i,0]>0.5]
        obj=float(lamb*sum([abs(tt[i,0]) for i in supp]))
        obj1=0.0
        theta=tt[supp,0]
        XX=np.matrix(X[:,supp])
        t=np.diag(1.0-2.0*y)*XX*theta
        t1=np.log(1.0+np.exp(t))
        obj1+=float(sum(sum(t1)))
        obj+=float(sum(sum(t1))/n)
        gradtt=np.matrix([0.0]*d).reshape((d, 1))
        gradz=np.matrix([0.0]*d).reshape((d, 1))
        t2=1.0/(1.0+np.exp(-t))
        tt1=(np.diagflat(t2)*np.diag(1.0-2.0*y)*X).transpose()
        gradtt+=np.matrix(tt1.sum(axis=1))    
        return gradz,gradtt,obj,obj1
    
    def lazycuts(M,where):
        if where == GRB.callback.MIPSOL:
            xsol = M.cbGetSolution(x)
            xsol=np.matrix([xsol[j]for j in range(n)]).transpose()            
            grad, f0=Grad(xsol)
            for i in range(p):
                M.cbLazy(f0[i,0]+quicksum(grad[i,j]*(x[j]-xsol[j]) for j in range(n))<=t[i])

    M = Model()
    bigM=min(n*np.log(2)/lamb,obj1/lamb)
    t = M.addMVar(1,  lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    theta  = M.addMVar(d, lb=-bigM, ub=bigM, vtype=GRB.CONTINUOUS)
    mu  = M.addMVar(d, lb=0, ub=bigM, vtype=GRB.CONTINUOUS)
    zvar  = M.addMVar(d, lb=0.0, ub=1, vtype=GRB.CONTINUOUS)

    M.setObjective((float(1.0/n))*t.sum()+lamb*mu.sum(), GRB.MINIMIZE)
    M.addConstr(zvar.sum()==kbar)
    M.addConstr(theta<=np.diag([bigM]*d)@zvar)
    M.addConstr(theta>=-np.diag([bigM]*d)@zvar)
    M.addConstr(mu<=np.diag([bigM]*d)@zvar)
    M.addConstr(mu>=theta)
    M.addConstr(mu>=-theta)
  
    zsol=[0]*d
    zsol=np.matrix(zsol).transpose()
    thetasol=[1.0]*d
    thetasol=np.matrix(thetasol).transpose()
    for i in range(kbar):
        zsol[i,0]=1.0    
    gradz,gradtt,ub1,f0=Grad(zsol,thetasol)
    const=np.array((gradtt.transpose()*thetasol))[0]
    gradz=np.array(gradz.transpose())[0]
    gradtt=np.array(gradtt.transpose())[0]
    f0=np.array([f0])
    M.addConstr(f0+gradtt @theta-const<=t)
    M.params.timelimit = 3600 
    start=time.time()
    
    iteration=0
    stepsize=0.005
    while (iteration<=-100) and time.time() - start<=3600:   
        zsol=[1]*d
        zsol=np.matrix(zsol).transpose()
        gradz,gradtt,ub1,f0=Grad(zsol,thetasol)
        const=np.array((gradtt.transpose()*thetasol))[0]
        gradz=np.array(gradz.transpose())[0]
        gradtt=np.array(gradtt.transpose())[0]
        f0=np.array([f0])
        M.addConstr(f0+gradtt @theta-const<=t)
        thetasol=thetasol-stepsize*np.matrix(gradtt).reshape((d, 1))
        iteration+=1
    zsol=[1]*d
    zsol=np.matrix(zsol).transpose()
    for i in range(d):
        if i>=kbar:
            zsol[i,0]=0
    gradz,gradtt,ub1,f0=Grad(zsol,thetasol)
    const=np.array((gradtt.transpose()*thetasol))[0]
    gradz=np.array(gradz.transpose())[0]
    gradtt=np.array(gradtt.transpose())[0]
    f0=np.array([f0])
    M.addConstr(f0+gradtt @theta-const<=t)
    maxiter=1000
    LB=0
    UB=1e30
    Gap=1.0
    iteration=0
    step=1.0
    M.params.OutputFlag = 0
    for j in range(d):
        zvar[j].vtype=GRB.BINARY 
    check=0
    while (iteration<=maxiter) and (UB-LB)/abs(UB)>1e-4 and time.time() - start<=3600:
        UB=min(UB,float(ub1))
        print("at iteration %d, the current upper bound is %f, current lower bound is %f" %(iteration, UB,LB))
        M.optimize()
        zsol=np.matrix([float(zvar[j].x) for j in range(d)]).transpose()
        zsol=zsol.reshape((d, 1))
        thetasol=np.matrix([float(theta[j].x) for j in range(d)]).transpose()
        thetasol=thetasol.reshape((d, 1))
        LB=M.objval
        gradz,gradtt,ub1,f0=Grad(zsol,thetasol)
        const=np.array((gradtt.transpose()*thetasol))[0]
        gradz=np.array(gradz.transpose())[0]
        gradtt=np.array(gradtt.transpose())[0]
        f0=np.array([f0])
        M.addConstr(f0+gradtt @theta-const<=t)        
        iteration+=1
    print("at iteration %d, the current upper bound is %f, current lower bound is %f" %(iteration, UB,LB))
    RTime=time.time() - start
    MTime = time.time() - start
    return M, zvar,theta,RTime, MTime,UB,LB


def loaddata(filename,pp):
    x, y = [], []
    Xy=pd.read_excel(filename+'.xlsx',sheet_name=filename, index_col=None)  
    y=Xy[Xy.columns[0]].values.tolist()
    x=Xy[Xy.columns[1:]].values.tolist()
    y=y[0:pp]
    x=x[0:pp]
    c=len(x[0])
    xnew=[]
    for i in range(len(x)):
        b=[]
        for j in range(c):    
            for k in range(j, c, 1):
                mm=x[i][j]*x[i][k] 
                b.append(mm)
        xnew.append(b)
    return np.array(xnew), np.array(y)

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

instance_number=0
for pp in range(5, 110, 5):
    x, y = loaddata('student',pp)
    X = x
    n, d = int(X.shape[0]), int(X.shape[1])
    kbar=20
    lamb=1e-2
    M, zvar, theta = logistic0(X,y,kbar,lamb,2,1,-2)
    M.params.timelimit = 10
    start0 = time.time()
    M.optimize()
    Mtime0 = time.time() - start0
    sa1,sa2,obj1 = Evaluation2(X,y,kbar,lamb,zvar,theta)
    M, zvar, theta,Rtime, Mtime , UB, LB = logistic(X,y,kbar,lamb,obj1)
    Gap = 1.0 - LB/(UB + 1e-7)
    Mtime_total = Mtime + Mtime0
    EUB = Evaluation(X,y,kbar,lamb,zvar,theta)
    Approx_Gap = abs(UB-EUB)/EUB
    df.loc[instance_number] = np.array([n,d,kbar, UB, LB, Rtime, Mtime, Mtime_total, Gap, EUB, Approx_Gap])
    df.to_csv('branch_cut_sparse_logistic_regression_student.csv')   
    instance_number = instance_number+1
        
    try:
        print([float(zvar[i].x) for i in range(d)])
        print([float(theta[i].x) for i in range(d)])
        print("The true obj is %f" %(Evaluation(X,y,kbar,lamb,zvar,theta)))
        
    except:
        print("Could not get solution")
        
