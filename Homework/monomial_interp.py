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

def driver1(): 

    f = lambda x: 1/ (1+(10*x)**2)
    
    N = 400
    a = -1
    b = 1
    
    h = 2/(N-1)
    ''' Create interpolation nodes'''
    # xint = np.linspace(a,b,N+1)
    # xint = []
    
    # for i in range(1,N+1):
    #     z = (-1 + (i-1)*h)
    #     xint.append(z)
    # xint = np.array(xint)
    
    xint = np.linspace(a,b,N+1)

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
    
#    print('coef = ', coef)

# No validate the code
    Neval = 100    
    xeval = np.linspace(a,b,Neval+1)
    yeval = eval_monomial(xeval,coef,N,Neval)

# exact function
    yex = f(xeval)
    
    err =  norm(yex-yeval) 
    print('err = ', err)
    
    plt.plot(xeval,yeval,"o")
    plt.plot(xeval,yex)
    return

def driver3(): 

    f = lambda x: 1/ (1+(10*x)**2)
    
    N = 400
    a = -1
    b = 1
    
    h = 2/(N-1)
    ''' Create interpolation nodes'''
    # xint = np.linspace(a,b,N+1)
    # xint = []
    
    # for i in range(1,N+1):
    #     z = (-1 + (i-1)*h)
    #     xint.append(z)
    # xint = np.array(xint)
    xint = Chebyshev(N+1)
    
    # xint = np.linspace(a,b,N+1)

#    print('xint =',xint)
    '''Create interpolation data'''
    yint = f(xint)
#    print('yint =',yint)
    
    ''' Create the Vandermonde matrix'''
    V = Vandermonde(xint,N)
    print("V is", V)

    ''' Invert the Vandermonde matrix'''    
    Vinv = inv(V)
#    print('Vinv = ' , Vinv)
    
    ''' Apply inverse to rhs'''
    ''' to create the coefficients'''
    coef = Vinv @ yint
    
#    print('coef = ', coef)

# No validate the code
    Neval = 100    
    xeval = np.linspace(a,b,Neval+1)
    yeval = eval_monomial(xeval,coef,N,Neval)

# exact function
    yex = f(xeval)
    
    err =  abs(yex-yeval) 
    print('err = ', err)
    
    plt.plot(xeval,yeval,"o")
    plt.plot(xeval,yex)
    plt.legend(labels=("Monomial","Function","Chebyshev"))
    plt.title(("Comparison of point distributors, order n=",N))
    
    plt.figure()
    plt.semilogy(xeval,err,"go")
    plt.title(("Monomial Error, order n=",N))
    
    return

def driver2():

    f = lambda x: 1./(1.+(10*x)**2)
    # fp = lambda x: -2*x/(1.+x**2)**2

    N = 400
    ''' interval'''
    a = -1
    b = 1
   
    ''' create equispaced interpolation nodes'''
    xint = np.linspace(a,b,N+1)
    
    ''' create interpolation data'''
    yint = np.zeros(N+1)
    # ypint = np.zeros(N+1)
    for jj in range(N+1):
        yint[jj] = f(xint[jj])
        # ypint[jj] = fp(xint[jj])
    
    ''' create points for evaluating the Lagrange interpolating polynomial'''
    Neval = 1000
    xeval = np.linspace(a,b,Neval+1)
    yevalL = np.zeros(Neval+1)
    # yevalH = np.zeros(Neval+1)
    for kk in range(Neval+1):
      yevalL[kk] = eval_lagrange(xeval[kk],xint,yint,N)
      # yevalH[kk] = eval_hermite(xeval[kk],xint,yint,ypint,N)

    ''' create vector with exact values'''
    fex = np.zeros(Neval+1)
    for kk in range(Neval+1):
        fex[kk] = f(xeval[kk])
    
    
    plt.figure()
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yevalL,'bs--',label='Lagrange') 
    # plt.plot(xeval,yevalH,'c.--',label='Hermite')
    plt.title("Lagrange Interpolation")
    plt.semilogy()
    plt.show()
         
    errL = abs(yevalL-fex)
    # errH = abs(yevalH-fex)
    plt.figure()
    plt.semilogy(xeval,errL,'bs--',label='Lagrange')
    # plt.semilogy(xeval,errH,'c.--',label='Hermite')
    plt.title(("Lagrange Error, n=",N))
    plt.show()            


def eval_hermite(xeval,xint,yint,ypint,N):

    ''' Evaluate all Lagrange polynomials'''

    lj = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    ''' Construct the l_j'(x_j)'''
    lpj = np.zeros(N+1)
#    lpj2 = np.ones(N+1)
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
#              lpj2[count] = lpj2[count]*(xint[count] - xint[jj])
              lpj[count] = lpj[count]+ 1./(xint[count] - xint[jj])
              

    yeval = 0.
    
    for jj in range(N+1):
       Qj = (1.-2.*(xeval-xint[jj])*lpj[jj])*lj[jj]**2
       Rj = (xeval-xint[jj])*lj[jj]**2
#       if (jj == 0):
#         print(Qj)
         
#         print(Rj)
#         print(Qj)
#         print(xeval)
 #        return
       yeval = yeval + yint[jj]*Qj+ypint[jj]*Rj
       
    return(yeval)
       


def eval_lagrange(xeval,xint,yint,N):

    lj = np.ones(N+1)
    
    for count in range(N+1):
       for jj in range(N+1):
           if (jj != count):
              lj[count] = lj[count]*(xeval - xint[jj])/(xint[count]-xint[jj])

    yeval = 0.
    
    for jj in range(N+1):
       yeval = yeval + yint[jj]*lj[jj]
  
    return(yeval)
  

def  eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    
#    print('yeval = ', yeval)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
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

def Chebyshev(n):
    K = np.arange(1,n+1)
    points = np.cos((2*K-1)*np.pi/(2*n))
    return points

driver1()
driver3()
driver2()