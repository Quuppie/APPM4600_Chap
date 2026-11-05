"""
 This script is useful for understanding the performance of the 
 cubic spline.  It includes implementations of both clamped and 
 natural splines.  It also has the option to change the nodes from equispaced
 to chebychev nodes to see if that changes the performance. 
 Also test changing the number of nodes.
 There are two example functions.  The performance changes.
 Why?
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



import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 
from numpy.linalg import norm


def driver():

# example 1    
    f = lambda x: np.exp(x)
    fp = lambda x: np.exp(x)
    a = 0
    b = 1

# example 2
#    f = lambda x: 1/(1 +(5*x)**2)
#    fp = lambda x: -2*5*x/(1 +(5*x)**2)**2   
#    a = -5
#    b = 5
    
    
    ''' number of intervals'''
    Nint = 10
    xint2 = np.linspace(a,b,Nint+1)
# create chebychev nodes
#    xint = np.zeros(Nint+1)
#    for j in range(1,Nint+2):
#       xint[j-1] = np.cos(np.pi*(2*j-1)/(2*(Nint+1)))
# scale  for the interval
#    m = (b-a)/2 
#    c = (a+b)/2
#    xint = m*xint+c
#    xint = xint[::-1]
    xint = xint2
    yint = f(xint)
    yint2 = f(xint2)
    yp_b = np.zeros(2)
    yp_b[0] = fp(xint[0])
    yp_b[1] = fp(xint[Nint])
    
    ''' create points you want to evaluate at'''
    Neval = 100
    xeval =  np.linspace(xint[0],xint[Nint],Neval+1)
    
    
    
    (M,C,D) = create_natural_spline(yint2,xint2,Nint)
    (Mc,Cc,Dc) = create_clamped_spline(yint,yp_b, xint,Nint)    
    
    yeval = eval_cubic_spline(xeval,Neval,xint2,Nint,M,C,D)
    yevalc = eval_cubic_spline(xeval,Neval,xint,Nint,Mc,Cc,Dc)
    
#    print('yeval = ', yevalc)
    
    ''' evaluate f at the evaluation points'''
    fex = f(xeval)
        
    nerr = norm(fex-yeval)
    print('nerr = ', nerr)
    
    plt.figure()    
    plt.plot(xeval,fex,'ro-',label='exact function')
    plt.plot(xeval,yeval,'bs--',label='natural spline') 
    plt.plot(xeval,yevalc,'gs--',label='clamped spline') 
    plt.legend
    plt.show()
     
    err = abs(yeval-fex)
    errc = abs(yevalc-fex)
    plt.figure() 
    plt.semilogy(xeval,err,'ro--',label='natural spline')
    plt.semilogy(xeval,errc,'bo--',label='clamped spline')
    plt.legend()
    plt.show()
    
def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = 1.0
    for j in range(1,N):
       A[j][j-1] = h[j-1]/6
       A[j][j] = (h[j]+h[j-1])/3 
       A[j][j+1] = h[j]/6
    A[N][N] = 1

    Ainv = inv(A)
    
    M  = Ainv.dot(b)

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)

def create_clamped_spline(yint,yp_b, xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip
    b[0]  = -yp_b[0] +(yint[1]-yint[0])/h[0]
    b[N] = -yp_b[1] +(yint[N]-yint[N-1])/h[N-1]

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = h[0]/3.
    A[0][1] = h[0]/6.
    for j in range(1,N):
       A[j][j-1] = h[j-1]/6
       A[j][j] = (h[j]+h[j-1])/3 
       A[j][j+1] = h[j]/6
    A[N][N] = h[N-1]/3
    A[N][N-1] = h[N-1]/6

    Ainv = inv(A)
    
    M  = Ainv.dot(b)

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)


       
def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
    yeval = (Mi*(xip-xeval)**3 +(xeval-xi)**3*Mip)/(6*hi) \
            + C*(xip-xeval) + D*(xeval-xi)
    return yeval 
    
    
def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval+1)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]
#        print('atmp =', atmp)
#       print('btmp =', btmp)
#        print('ind =', ind)
#       print('xloc =', xloc)
#        return

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
#        print('yloc = ', yloc)
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)
           
driver()               

