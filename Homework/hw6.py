#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 18:00:53 2026

@author: groot
Code for HW6
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 

def driver1():
#%% P1
    evaltype = evalF
    jtype = evalJ1
    Nmax = 100
    tol = 10**-4
    x0 = {0: np.array([-0.5,0.25]), 1: np.array([-1.5,1.25]), 2: np.array([-3,3])}
    for i in range(3):
        [xstarf, distsf, ierf, itsf] = FixedPt(x0[i],tol,Nmax)
        print("Run",i+1)
        print("The fixed point is",xstarf)
        print("Fixed Error reads", ierf)
        print("Fixed took this many iterations", itsf)
        
        
        [xstarn,distsn, iern,itsn] = Newton(x0[i],evaltype,jtype,tol,Nmax)
        print("The root of Newton is",xstarn)
        print("Newton Error reads", iern)
        print("Newton took this many iterations:", itsn)
        

        plt.figure()
        plt.plot(range(Nmax),distsf,label="Fixed-Pt")
        plt.plot(range(Nmax),distsn,label="Newton")
        plt.yscale('log')
        plt.title("Error")
        plt.legend()
    return
def driver2():
#%% P2
    evaltype = evalF2
    jtype = evalJ2
    Nmax = 100
    tol = 10**-4
    dataout = dict({})
    x0 = {0: np.array([1,1]), 1: np.array([1,-1]), 2: np.array([0,0])}
    for i in range(3):#
        t = time.time()
        for j in range(1):
            [xstarn,distsn, iern,itsn] = Newton(x0[i],evaltype,jtype,tol,Nmax)
        elapsed = time.time()-t
        print("Newton Took this many seconds", elapsed/50)
        # print("P2: The root of Newton is",xstarn)
        # print("Newton Error reads", iern)
        # print("Newton took this many iterations:", itsn)
        
        t = time.time()
        for j in range(1):
            [xstarl,distsl, ierl,itsl] = LazyNewton(x0[i],evaltype,jtype,tol,Nmax)
        elapsed = time.time()-t
        print("Lazy Newton Took this many seconds", elapsed/50)
        # print("P2: The root of Lazy Newton is",xstarl)
        # print("Lazy Newton Error reads", ierl)
        # print("Lazy Newton took this many iterations:", itsl)
        
        t = time.time()
        for j in range(1):
            [xstarb,distsb, ierb,itsb] = Broyden(x0[i],evaltype,jtype,tol,Nmax)
        elapsed = time.time()-t
        print("Broyden Took this many seconds", elapsed/50)
        

        # print("P2: The root of Broyden is",xstarb)
        # print("Broyden Error reads", ierb)
        # print("Broyden took this many iterations:", itsb)
        dataout = {i:{"Newton Error":iern,"Lazy Newton Error":ierl,"Broyden Error":ierb,"Newton Iterations":itsn,"Lazy Iterations":itsl,"Broyden Iterations":itsb,"Newton Root":xstarn,"Lazy Root":xstarl,"Broyden Root":xstarb}}
        datatrueout = {**dataout}
        print("Run ",i,dataout)
    return
    
def driver3():
#%% P3 - use evalFS,evalJS
    print("Problem 3")
    evaltype = evalFS
    jtype = evalJS
    Nmax = 100
    tol = 10**-6
    
    x0 = {0: np.array([-0.1,0,0.1]), 1: np.array([1,1,0]), 2: np.array([0,0,5])}
    
    for i in range(3):
        t = time.time()
        for j in range(50):
            print("------>> Steep Run",i)
            [xstarn,distsn, iern,itsn] = Newton(x0[i],evaltype,jtype,tol,Nmax)
            # print("The root of Newton is",xstarn)
            # print("Newton Error reads", iern)
            # print("Newton took this many iterations:", itsn)
        elapsed=time.time()-t
        print("Total time taken Newton:",elapsed/50)
        t=time.time()
        for j in range(50):
            [xstar,gval,dists,its,ier] = SteepestDescent(x0[i],tol,Nmax)
            # print("the steepest descent code found the solution ",xstar)
            # print("g evaluated at this point is ", gval)
            # print("Steepest Descent took this many steps",its)
            # print("ier is ", ier)
        elapsed=time.time()-t
        print("Total time Steepest:",elapsed/50)
        t = time.time()
        for j in range(50):
            tol = 5*10**-2
            [xs0,gval,dists,its,ier] = SteepestDescent(x0[i],tol,Nmax)
            # print("Stage 1: Steepest Descent reached",xs0)
            # print("Stage 1: Steepest Descent took this many steps",its)
            # print("Stage 1 G is",gval)
            # print("Stage 1 error reads",ier)
            tol = 10**-6
            [xstarn,distsn, iern,itsn] = Newton(x0[i],evaltype,jtype,tol,Nmax)
            # print("Stage 2: Newton reached",xstarn)
            # print("Stage 2 iterations were",itsn)
            # print("Stage 2 error reads",ier)
            # print("Total iterations were",itsn+its)
            
        elapsed = time.time()-t
        print("Total time taken:",elapsed/50)
        
       
    
    return
#%% Funcs

def evalF(x):
    # vector function that you want to find the roots of
    F = np.zeros(2)
    F[0] = 3*x[0]**2+4*x[1]**2-1
    F[1] = x[1]**3-8*x[0]**3-1
    return F

def evalF1(x):
    # vector function that you want to find the roots of
    F = np.zeros(2)
    F[0] = 3*x[0]**2+4*x[1]**2-1
    F[1] = x[1]**3-8*x[0]**3-1
    return F

def evalF2(x):
    # vector function that you want to find the roots of
    F = np.zeros(2)
    F[0] = x[0]**2 + x[1]**2 - 4
    F[1] = np.e**(x[0]) + x[1] - 1
 
    return F



def evalFFixed(x): 

    F = np.zeros(2)
    
    # F[0] = x[0]-(x[0]**2+x[1]**2-5)/4
    # F[1] = x[1]-(x[0]*x[1]-2)/2
    F[0] = x[0] - (0.016 * (3*x[0]**2+4*x[1]**2-1) - 0.17 * (x[1]**3-8*x[0]**3-1))
    F[1] = x[1] - (0.52 * (3*x[0]**2+4*x[1]**2-1) -  0.26 * (x[1]**3-8*x[0]**3-1))
 
    return F

def evalJ1(x):
    # Jacobian of the vector function you want to find the roots of
    
    J = np.array([[6*x[0],8*x[1]],[-24*x[0]**2,3*x[1]**2]])
    
    return J


def evalJ2(x):
    J = np.array([[2*x[0],2*x[1]],[np.e**x[0],1]])
    
    return J

def evalJ3(x):
    J = np.array([
        [1-x[1]*x[2]*np.sin(x[0]*x[1]*x[2]),-x[0]*x[2]*np.sin(x[0]*x[1]*x[2]),-x[0]*x[1]*np.sin(x[0]*x[1]*x[2])],
        [-1/4*(1-x[0])**(-3/4),1,0.1*x[2]-0.15],
        [-2*x[0],-0.2*x[1]+0.1,1]]
        )
    
    return J

#functions:
def evalFS(x): 

    F = np.zeros(3)
    F[0] = x[0] + np.cos(x[0]*x[1]*x[2]) - 1
    F[1] = ((1-x[0])**0.25) + x[1] + 0.05*x[2]**2 - 0.15*x[2] - 1
    F[2] = -x[0]**2 - 0.1*x[1]**2 + 0.01*x[1] + x[2] - 1
 
    return F

def evalJS(x): 
# Jacobian of the vector valued function you are trying to find the root of.

    J =np.array([[1.+x[1]*x[2]*math.sin(x[0]*x[1]*x[2]),x[0]*x[2]*math.sin(x[0]*x[1]*x[2]),x[1]*x[0]*math.sin(x[0]*x[1]*x[2])],
          [-0.25*(1-x[0])**(-0.75),1,0.1*x[2]-0.15],
          [-2*x[0],-0.2*x[1]+0.01,1]])
    return J

def evalg(x):
# The least squares function you are minimizing

    F = evalFS(x)
    g = F[0]**2 + F[1]**2 + F[2]**2
    return g

def eval_gradg(x):
# Grad g
    F = evalFS(x)
    J = evalJS(x)
    
    gradg = np.transpose(J).dot(F)
    return gradg


###############################
### steepest descent code

def SteepestDescent(x,tol,Nmax):
    dists = np.zeros(Nmax)
    for its in range(Nmax):
        g1 = evalg(x)
        z = eval_gradg(x)
        z0 = norm(z)

        if z0 == 0:
            print("zero gradient")
        z = z/z0
        alpha1 = 0
        alpha3 = 1
        dif_vec = x - alpha3*z
        g3 = evalg(dif_vec)
        
        dists[its] = norm(dif_vec)
        
        while g3>=g1:
            alpha3 = alpha3/2
            dif_vec = x - alpha3*z
            g3 = evalg(dif_vec)
            
        if alpha3<tol:
            print("no likely improvement")
            ier = 0
            return [x,g1,dists,its,ier]
        
        alpha2 = alpha3/2
        dif_vec = x - alpha2*z
        g2 = evalg(dif_vec)

        h1 = (g2 - g1)/alpha2
        h2 = (g3-g2)/(alpha3-alpha2)
        h3 = (h2-h1)/alpha3

        alpha0 = 0.5*(alpha2 - h1/h3)
        dif_vec = x - alpha0*z
        g0 = evalg(dif_vec)
        
        if g0<=g3:
            alpha = alpha0
            gval = g0

        else:
            alpha = alpha3
            gval =g3

        x = x - alpha*z

        if abs(gval - g1)<tol:
            ier = 0
            return [x,gval,dists,its,ier]

    print('max iterations exceeded')    
    ier = 1        
    return [x,g1,dists,its,ier]


def FixedPt(x0,tol,Nmax):   #not updated to the evaltype
    dists = np.zeros(Nmax)
    xstar = np.zeros(Nmax)
    for its in range(Nmax):
        x1 = evalFFixed(x0)
        # x1 = np.array([[x0[0]],[x0[1]]]) - np.array([[0.016, -0.17],[0.52,-0.26]]) @ np.transpose(evalF(x0))
        if norm(x1-x0)<tol:
            xstar = x1
            # dists = dists[0:its+1]
            ier = 0
            return[xstar,dists,ier,its]
        
        dists[its] = np.linalg.norm(x1-x0)
        x0 = x1
        
    xstar=x1
    ier = 1
    return[xstar,dists,ier,its]

def Newton(x0,evaltype,jtype,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    dists = np.zeros(Nmax)
    xstar = np.zeros(Nmax)
    for its in range(Nmax):
        J = jtype(x0)
           
        try:
            Jinv = inv(J)
            
            F = evaltype(x0)
            
            x1 = x0 - Jinv @ F
            dists[its] = np.linalg.norm(F-evaltype(x1))
            
            if (norm(x1-x0) < tol):
                xstar = x1
                ier = 0
                return[xstar, dists, ier, its]
            
            x0 = x1
            
        except np.linalg.LinAlgError as err:
            x1 = np.nan
            if "Singular" in str(err):
                Jinv = np.nan
                print("WARNING: J is singular, exiting loop")

            else:
                print("No idea what's wrong mate")
           
                     
       # F = evalF(x0,opt)
       
       # x1 = x0 - Jinv @ F
       # dists[its] = np.linalg.norm(F-evalF(x1,opt))
       
       # if (norm(x1-x0) < tol):
       #     xstar = x1
       #     ier = 0
       #     return[xstar, dists, ier, its]
       
       # x0 = x1

    xstar = x1
    ier = 1
    return[xstar,dists,ier,its]

def Newton3(x0,evaltype,jtype,tol,Nmax,opt):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    dists = np.zeros(Nmax)
    xstar = np.zeros(Nmax)
    for its in range(Nmax):
        J = jtype(x0)

        try:
            Jinv = inv(J)
            
            F = evaltype(x0,opt)
            
            x1 = x0 - Jinv @ F
            dists[its] = np.linalg.norm(F-evaltype(x1,opt))
            
            if (norm(x1-x0) < tol):
                xstar = x1
                ier = 0
                return[xstar, dists, ier, its]
            
            x0 = x1
            
        except np.linalg.LinAlgError as err:
            x1 = np.nan
            if "Singular" in str(err):
                Jinv = np.nan
                print("WARNING: J is singular, exiting loop")
                break
            else:
                print("No idea what's wrong mate")
                break
           
                     
       # F = evalF(x0,opt)
       
       # x1 = x0 - Jinv @ F
       # dists[its] = np.linalg.norm(F-evalF(x1,opt))
       
       # if (norm(x1-x0) < tol):
       #     xstar = x1
       #     ier = 0
       #     return[xstar, dists, ier, its]
       
       # x0 = x1

    xstar = x1
    ier = 1
    return[xstar,dists,ier,its]

def LazyNewton(x0,evaltype,jtype,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    dists = np.zeros(Nmax)
    J = jtype(x0)
    
    try: 
        Jinv = inv(J)
        for its in range(Nmax):
    
           F = evaltype(x0)
           x1 = x0 - Jinv @ F
           dists[its] = np.linalg.norm(F-evaltype(x1))
           if (norm(x1-x0) < tol):
               xstar = x1
               ier = 0
               return[xstar,dists,ier,its]
               
           x0 = x1
        
    except np.linalg.LinAlgError as err:
        its = 0
        x1 = np.nan
        if "Singular" in str(err):
            Jinv = np.nan
            print("WARNING: J is singular, exiting loop")
            
        else:
            print("No idea what's wrong mate")
            
            
    xstar = x1
    ier = 1
    return[xstar,dists,ier,its]   
    
def Broyden(x0,evaltype,jtype,tol,Nmax):
    '''tol = desired accuracy
    Nmax = max number of iterations'''

    '''Sherman-Morrison 
   (A+xy^T)^{-1} = A^{-1}-1/p*(A^{-1}xy^TA^{-1})
    where p = 1+y^TA^{-1}Ax'''

    '''In Newton
    x_k+1 = xk -(G(x_k))^{-1}*F(x_k)'''


    '''In Broyden 
    x = [F(xk)-F(xk-1)-\hat{G}_k-1(xk-xk-1)
    y = x_k-x_k-1/||x_k-x_k-1||^2'''

    ''' implemented as in equation (10.16) on page 650 of text'''
    
    '''initialize with 1 newton step'''
    dists = np.zeros(Nmax)
    try: 
        A0 = jtype(x0)
    
        v = evaltype(x0)
        A = np.linalg.inv(A0)
    
        s = -A.dot(v)
        xk = x0+s
        x1=x0
        for  its in range(Nmax):
           '''(save v from previous step)'''
           w = v
           ''' create new v'''
           v = evaltype(xk)
           '''y_k = F(xk)-F(xk-1)'''
           y = v-w;                   
           '''-A_{k-1}^{-1}y_k'''
           z = -A.dot(y)
           ''' p = s_k^tA_{k-1}^{-1}y_k'''
           p = -np.dot(s,z)                 
           u = np.dot(s,A) 
           ''' A = A_k^{-1} via Morrison formula'''
           tmp = s+z
           tmp2 = np.outer(tmp,u)
           A = A+1./p*tmp2
           ''' -A_k^{-1}F(x_k)'''
           s = -A @ v
           xk = xk+s
           dists[its] = np.linalg.norm(xk-x1)
           x1=xk
           if (norm(s)<tol):
              alpha = xk
              ier = 0
              return[alpha,dists,ier,its]
      
    except np.linalg.LinAlgError as err:
        xk = np.nan
        its = 0
        if "Singular" in str(err):
            A = np.nan
            print("WARNING: A is singular, exiting loop")
        else:
            print("No idea what's wrong mate")
    alpha = xk
    ier = 1
    return[alpha,dists,ier,its]

# driver1()

# driver2()

driver3()