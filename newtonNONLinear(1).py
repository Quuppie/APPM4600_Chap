"""
 This compares Newton's method, Broyden and Lazy Newton for 
 computing the roots of vector valued functions.
 The function and the Jacobian are stored in subroutines and need 
 to be changed for different problems.  
 
"""

############################################# 
"""
Copyright (C) 2025  Adrianna M. Gillman

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
############################################# 


import numpy as np
import math
import time
from numpy.linalg import inv 
from numpy.linalg import norm 
import matplotlib.pyplot as plt

def driver():

    x0 = np.array([1.0, 1.0])
    
    Nmax = 100
    tol = 1e-10
    
    t = time.time()
    for j in range(50):
      [xstar,ndists,ier,nits] =  Newton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Newton: the error message reads:',ier) 
    print('Newton: took this many seconds:',elapsed/50)
    print('Netwon: number of iterations is:',nits)
     
    
    t = time.time()
    for j in range(20):
      [xstar,ldists,ier,lits] =  LazyNewton(x0,tol,Nmax)
    elapsed = time.time()-t
    print(xstar)
    print('Lazy Newton: the error message reads:',ier)
    print('Lazy Newton: took this many seconds:',elapsed/20)
    print('Lazy Newton: number of iterations is:',lits)
     
    t = time.time()
    for j in range(20):
      [xstar,bdists,ier,bits] = Broyden(x0, tol,Nmax)     
    elapsed = time.time()-t
    print(xstar)
    print('Broyden: the error message reads:',ier)
    print('Broyden: took this many seconds:',elapsed/20)
    print('Broyden: number of iterations is:',bits)
    
    plt.figure()
    plt.plot(range(Nmax),ndists,label="Newton")
    plt.plot(range(Nmax),ldists,label="Lazy")
    plt.plot(range(Nmax),bdists,label="Broyden")
    plt.xlim([0,1.5*bits])
    plt.title("Error versus Iteration")
    plt.legend()
    plt.savefig("hw6p1Iteration",dpi=300)
    
#Problem 3    
    plt.figure()
    x0 = np.array([1,1,1])
    dists = specialdim(x0,tol,Nmax)
    print(dists)
    plt.plot(range(Nmax),ndists,label="Whack")
    plt.title("Error vs Iteration for Ellipsoid")
    plt.savefig("hw6p3Fancy",dpi=300)
    
def evalF(x): 
# vector function that you want to find the roots of

    F = np.zeros(2)
    
    F[0] = 3*x[0]**2 - x[1]**2
    F[1] = 3*x[0]*x[1]**2 - x[0]**3 - 1
 
    return F
    
def evalEllipse(x):
    E = x[0]**2+4*x[1]**2+4*x[2]**2-16
    return E

def specialdim(x0,tol,Nmax):
    dists = np.zeros(Nmax)
    d = np.array([2*x0[0],8*x0[1],8*x0[2]])
    for its in range(Nmax):
        x1 = x0 - evalEllipse(x0)
        dists[its] = np.linalg.norm(evalF(x1)-evalF(x0))
    return dists
    
def evalJ(x): 
# Jacobian of the vector function you want to find the roots of
    
    J = np.array([[6*x[0],-2*x[1]],[3*x[1]**2-3*x[0]**2,6*x[0]*x[1]]])

    return J


def Newton(x0,tol,Nmax):

    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    dists = np.zeros(Nmax)
    for its in range(Nmax):
       J = evalJ(x0)
       Jinv = inv(J)
       F = evalF(x0)
       
       x1 = x0 - Jinv @ F
       dists[its] = np.linalg.norm(F-evalF(x1))
       
       if (norm(x1-x0) < tol):
           xstar = x1
           ier = 0
           return[xstar, dists, ier, its]
       
       x0 = x1

    xstar = x1
    ier = 1
    return[xstar,dists,ier,its]

def LazyNewton(x0,tol,Nmax):

    ''' Lazy Newton = use only the inverse of the Jacobian for initial guess'''
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''
    dists = np.zeros(Nmax)
    J = evalJ(x0)
    Jinv = inv(J)
    for its in range(Nmax):

       F = evalF(x0)
       x1 = x0 - Jinv @ F
       dists[its] = np.linalg.norm(F-evalF(x1))
       if (norm(x1-x0) < tol):
           xstar = x1
           ier = 0
           return[xstar,dists,ier,its]
           
       x0 = x1
    
    xstar = x1
    ier = 1
    return[xstar,dists,ier,its]   
    
def Broyden(x0,tol,Nmax):
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
    A0 = evalJ(x0)

    v = evalF(x0)
    A = np.linalg.inv(A0)

    s = -A.dot(v)
    xk = x0+s
    x1=x0
    for  its in range(Nmax):
       '''(save v from previous step)'''
       w = v
       ''' create new v'''
       v = evalF(xk)
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
    alpha = xk
    ier = 1
    return[alpha,dists,ier,its]
     
        
if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()    
        
