#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:45:14 2026

@author: rootg
HW3 
"""

#Import
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import time

def driver():
    print("Bungis")
    ti = 20 #c
    ts = -15 #c
    alpha = 0.138*10**-6 #m^2/s
    tol = 10**-13
    tf = 5.184*10**6
    x0 = 0
    xbar=10
    Nmax=10000
    f = lambda x: (ti-ts)*(2/np.sqrt(np.pi))*sp.erf(x/(2*np.sqrt(2*alpha*tf))) + ts
    fprime = lambda x: (ti-ts)*(2/np.sqrt(np.pi))*np.exp((x/(2*np.sqrt(alpha*tf))))
    
    x = np.linspace(0,10,500)
    plt.plot(x,f(x))
    plt.grid('major')
    plt.title("Temperature vs depth at t=60 days")
    plt.xlabel("Depth (- m)")
    plt.ylabel("Temperature (c)")
    
    
#Bisection
    [astar, ier, count] = bisection(f,0,xbar,tol)
    
    print('Bisection')
    print('the approximate root is',astar)
    print(f'Bisection took {count} steps')
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))
    
#Newton 0.01
    [p,pstar,info,it] = newton(f,fprime,0.01,tol,Nmax)
    print('Newton x0=0.01')
    print('the approximate root is',pstar)
    print(f'Newton took {it} steps')
    print('the error message reads:',info)
    print('f(astar) =', f(pstar))
    
#Newton xbar
    [px,pstarx,infox,itx] = newton(f,fprime,xbar,tol,Nmax)
    print('Newton x0 = xbar')
    print('the approximate root is',pstarx)
    print(f'Newton took {itx} steps')
    print('the error message reads:',infox)
    print('f(astar) =', f(pstarx))
    

#Fixed point
    
#fixed point is alpha2 = 3.09... 
    fa = lambda x: x*pow((1+(7-pow(x,5)/(pow(x,2)))),3)
    fa = lambda x: x-(pow(x,5)-7)/pow(x,2)
    fa = lambda x: x-(pow(x,5)-7)/(5*pow(x,4))
    
    Nmax = 100
    tol = 1e-10

# test f1 '''
    x0 = 1.0
    [xstar1,ier] = fixedpt(fa,x0,tol,Nmax)
    print('For fa the approximate fixed point is:',xstar1)
    print('f1(xstar):',fa(xstar1))
    print('Error message reads:',ier)
    
    return
#%% Functions
def newton(f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """
  p = np.zeros(Nmax+1);
  p[0] = p0
  for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,it]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,it]
        
def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
    count = 0
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier,count]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier,count]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier,count]

    count = 0
    d = 0.5*(a+b)
    while (abs(fa-fb) >= tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, count]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
      astar = d
#      print('abs(d-a) = ', abs(d-a))

    ier = 0
    return [astar, ier, count]

def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    
    count = 0
    xstar = np.zeros((Nmax+1,1))
    xstar[0] = x0
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar[count] = x1
          ier = 0
          return [xstar,ier]
       x0 = x1
       xstar[count] = x1

    ier = 1
    return [xstar, ier]

driver()