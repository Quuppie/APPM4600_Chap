"""
 This script investigates the use of the monomial basis for interpolation.
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
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sympy as sp

def driver1(): # Problem 1

    f = lambda x: 1/(1+(10*x)**2)

    N = 10
    a = 0
    b = 1
    
    ''' Create interpolation nodes'''
    xint = []
    for i in range(1,N+2):
        z = (-1 + (i-1)*(2/(N-1)))
        xint.append(z)
    xint = np.array(xint)
    # xint = np.linspace(a,b,N+1)
#    print('xint =',xint)
    '''Create interpolation data'''
    yint = f(xint)
#    print('yint =',yint)
    
    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint,N)
#    print('V = ',V)

    ''' Invert the Vandermonde matrix'''    
    Vinv = inv(V)
#    print('Vinv = ' , Vinv)
    
    ''' Apply inverse to rhs'''
    ''' to create the coefficients'''
    coef = Vinv @ yint
    
    print('coef = ', coef)

# No validate the code
    Neval = 100
    xeval = np.linspace(a,b,Neval)
    xfine = np.linspace(a,b,Neval+1)
    yeval = eval_monomial(xeval,coef,N,Neval)

# exact function
    yex = f(xfine)
    
    plt.plot(xfine,yex) #actual
    plt.plot(xeval,yeval,'o') #interp
    plt.title("Monomial Basis Interpolation")
    plt.xlim([-2,2])
    # plt.ylim([-1,5])
    
    plt.show()
    
    ycomp = f(xeval)
    err =  abs(ycomp-yeval)

    plt.plot(xeval,err,'o')
    plt.title("Error")
    
    print(V)
    print(Vinv)

def  eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval)
    
#    print('yeval = ', yeval)
    
    for j in range(1,N+1):
      for i in range(Neval):
#        print('yeval[i] = ', yeval[i])
#        print('a[j] = ', a[j])
#        print('i = ', i)
#        print('xeval[i] = ', xeval[i])
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval

def Vandermonde(xint,N):

    V = np.zeros((N+1,N+1))
    
    ''' fill the first column'''
    for j in range(N+1):
       V[j][0] = 1.0

    for i in range(1,N+1):
        for j in range(N+1):
           V[j][i] = xint[j]**i

    return V     

driver1()    
