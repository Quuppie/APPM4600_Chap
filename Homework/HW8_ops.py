"""
 This script goes for HW8
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
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm
import math

def driverN():

    f = lambda x: np.e**x
    fp = lambda x: np.e**x
    # f = lambda x: np.sin(x)
    # fp = lambda x: np.cos(x)
    
    for i in range(1,5):
        N = 2
        ''' interval'''
        a = 0
        b = 1
       
        ''' create equispaced interpolation nodes'''
        xint = np.linspace(a,b,N+1)
        
        ''' create interpolation data'''
        yint = np.zeros(N+1)
        ypint = np.zeros(N+1)
        for jj in range(N+1):
            yint[jj] = f(xint[jj])
            ypint[jj] = fp(xint[jj])
        
        ''' create points for evaluating the Lagrange interpolating polynomial'''
        Neval = 1000
        xeval = np.linspace(a,b,Neval+1)
        yevalL = np.zeros(Neval+1)
        yevalH = np.zeros(Neval+1)
        for kk in range(Neval+1):
          yevalL[kk] = eval_lagrange(xeval[kk],xint,yint,N)
          yevalH[kk] = eval_hermite(xeval[kk],xint,yint,ypint,N)
          
        # create cubic splines
        
        yp_b = np.zeros(2)
        yp_b[0] = fp(xint[0])
        yp_b[1] = fp(xint[N])
        
        # Natural
        (Mn,Cn,Dn) = create_natural_spline(yint,xint,N)
        yevalN = eval_cubic_spline(xeval,Neval,xint,N,Mn,Cn,Dn)
        
        #Clamped
        (Mc,Cc,Dc) = create_clamped_spline(yint,yp_b,xint,N)
        yevalC = eval_cubic_spline(xeval,Neval,xint,N,Mc,Cc,Dc)
        
    
        ''' create vector with exact values'''
        fex = np.zeros(Neval+1)
        for kk in range(Neval+1):
            fex[kk] = f(xeval[kk])
        
        fig,axs = plt.subplots(4,2,figsize=(15,20),sharey="row")
        axs[0,0].set_ylim([-1,2])
        axs[0,0].plot(xeval,fex,'ro-')
        axs[0,0].plot(xeval,yevalL,'bs--',label='Lagrange') 
        axs[0,0].plot(xeval,yevalH,'c.--',label='Hermite')
        axs[0,0].set_title(str("Lagrange Polynomial, n =" + str(N)))
        axs[0,0].legend()
        # axs[0,0].semilogy()
        # plt.show()
             
        errL = abs(yevalL-fex)
        errH = abs(yevalH-fex)
        axs[1,0].semilogy(xeval,errL,'bs--',label='Lagrange Error')
        axs[1,0].semilogy(xeval,errH,'c.--',label='Hermite Error')
        axs[1,0].set_title(str("Approximation Error, n =" + str(N)))
        axs[1,0].legend()
        fig.show()  

    #splines
        axs[0,1].set_ylim([-1,2])
        axs[0,1].plot(xeval,fex,'ro-')
        axs[0,1].plot(xeval,yevalN,'g--',label='Natural') 
        axs[0,1].plot(xeval,yevalC,'y.--',label='Clamped')
        axs[0,1].set_title(("Splines, n =" + str(N)))
        axs[0,1].legend()
        # axs[0,1].semilogy()
             
        errN = abs(yevalN-fex)
        errC = abs(yevalC-fex)
        
        axs[1,1].semilogy(xeval,errN,'g--',label='Natural Error')
        axs[1,1].semilogy(xeval,errC,'y.--',label='Clamped Error')
        axs[1,1].set_title(str("Spline Error, n =" + str(N)))
        axs[1,1].legend()
        # fig.suptitle("Interpolation Equispaced")
        # fig.show()  
        #%% Boogying with the Chebyshev Nodes
        xint = np.zeros(N+1)
        for j in range(1,N+2):
           xint[j-1] = np.cos(np.pi*(2*j-1)/(2*(N+1)))
        # scale  for the interval
        m = (b-a)/2 
        c = (a+b)/2
        xint = m*xint+c
        xint = xint[::-1]
        ''' create interpolation data'''
        yint = np.zeros(N+1)
        ypint = np.zeros(N+1)
        for jj in range(N+1):
            yint[jj] = f(xint[jj])
            ypint[jj] = fp(xint[jj])
        
        ''' create points for evaluating the Lagrange interpolating polynomial'''
        Neval = 1000
        xeval = np.linspace(a,b,Neval+1)
        yevalL = np.zeros(Neval+1)
        yevalH = np.zeros(Neval+1)
        for kk in range(Neval+1):
          yevalL[kk] = eval_lagrange(xeval[kk],xint,yint,N)
          yevalH[kk] = eval_hermite(xeval[kk],xint,yint,ypint,N)
          
        # create cubic splines
        
        yp_b = np.zeros(2)
        yp_b[0] = fp(xint[0])
        yp_b[1] = fp(xint[N])
        
        # Natural
        (Mn,Cn,Dn) = create_natural_spline(yint,xint,N)
        yevalN = eval_cubic_spline(xeval,Neval,xint,N,Mn,Cn,Dn)
        
        #Clamped
        (Mc,Cc,Dc) = create_clamped_spline(yint,yp_b,xint,N)
        yevalC = eval_cubic_spline(xeval,Neval,xint,N,Mc,Cc,Dc)
        
    
        ''' create vector with exact values'''
        fex = np.zeros(Neval+1)
        for kk in range(Neval+1):
            fex[kk] = f(xeval[kk])
        
        # fig,axs = plt.subplots(2,2,figsize=(15,10),sharey="row")
        axs[2,0].set_ylim([-1,2])
        axs[2,0].plot(xeval,fex,'ro-')
        axs[2,0].plot(xeval,yevalL,'bs--',label='Lagrange') 
        axs[2,0].plot(xeval,yevalH,'c.--',label='Hermite')
        axs[2,0].set_title(str("Chebyshev Lagrange Polynomial, n =" + str(N)))
        axs[2,0].legend()
        # axs[2,0].semilogy()
        # plt.show()
             
        errL = abs(yevalL-fex)
        errH = abs(yevalH-fex)
        axs[3,0].semilogy(xeval,errL,'bs--',label='Lagrange Error')
        axs[3,0].semilogy(xeval,errH,'c.--',label='Hermite Error')
        axs[3,0].set_title(str("Chebyshev Approximation Error, n =" + str(N)))
        axs[3,0].legend()
        fig.show()  

    #splines
        axs[2,1].set_ylim([-1,2])
        axs[2,1].plot(xeval,fex,'ro-')
        axs[2,1].plot(xeval,yevalN,'g--',label='Natural') 
        axs[2,1].plot(xeval,yevalC,'y.--',label='Clamped')
        axs[2,1].set_title(("Chebyshev Splines, n =" + str(N)))
        axs[2,1].legend()
        # axs[2,1].semilogy()
             
        errN = abs(yevalN-fex)
        errC = abs(yevalC-fex)
        
        axs[3,1].semilogy(xeval,errN,'g--',label='Natural Error')
        axs[3,1].semilogy(xeval,errC,'y.--',label='Clamped Error')
        axs[3,1].set_title(str("Chebyshev Spline Error, n =" + str(N)))
        axs[3,1].legend()
        
        # fig.suptitle("Interpolation Chebyshev Nodes")
        
        fig.show()  
        
        #%% 
         
def driverS():

    f = lambda x: np.sin(10*x)
    fp = lambda x: 10*np.cos(x)
    
    for i in range(1,5):
        N = 5*i
        ''' interval'''
        a = -0
        b = 2*np.pi
       
        ''' create equispaced interpolation nodes'''
        xint = np.linspace(a,b,N+1)
        
        ''' create interpolation data'''
        yint = np.zeros(N+1)
        ypint = np.zeros(N+1)
        for jj in range(N+1):
            yint[jj] = f(xint[jj])
            ypint[jj] = fp(xint[jj])
        
        ''' create points for evaluating the Lagrange interpolating polynomial'''
        Neval = 1000
        xeval = np.linspace(a,b,Neval+1)
        yevalL = np.zeros(Neval+1)
        yevalH = np.zeros(Neval+1)
        for kk in range(Neval+1):
          yevalL[kk] = eval_lagrange(xeval[kk],xint,yint,N)
          yevalH[kk] = eval_hermite(xeval[kk],xint,yint,ypint,N)
          
        # create cubic splines
        
        yp_b = np.zeros(2)
        yp_b[0] = fp(xint[0])
        yp_b[1] = fp(xint[N])
        
        # Natural
        (Mn,Cn,Dn) = create_natural_spline(yint,xint,N)
        yevalN = eval_cubic_spline(xeval,Neval,xint,N,Mn,Cn,Dn)
        
        #Clamped
        (Mc,Cc,Dc) = create_clamped_spline(yint,yp_b,xint,N)
        yevalC = eval_cubic_spline(xeval,Neval,xint,N,Mc,Cc,Dc)
        
    
        ''' create vector with exact values'''
        fex = np.zeros(Neval+1)
        for kk in range(Neval+1):
            fex[kk] = f(xeval[kk])
        
        fig,axs = plt.subplots(4,2,figsize=(15,20),sharey="row")
        axs[0,0].set_ylim([-1,2])
        axs[0,0].plot(xeval,fex,'ro-')
        axs[0,0].plot(xeval,yevalL,'bs--',label='Lagrange') 
        axs[0,0].plot(xeval,yevalH,'c.--',label='Hermite')
        axs[0,0].set_title(str("Lagrange Polynomial, n =" + str(N)))
        axs[0,0].legend()
        # axs[0,0].semilogy()
        # plt.show()
             
        errL = abs(yevalL-fex)
        errH = abs(yevalH-fex)
        axs[1,0].semilogy(xeval,errL,'bs--',label='Lagrange Error')
        axs[1,0].semilogy(xeval,errH,'c.--',label='Hermite Error')
        axs[1,0].set_title(str("Approximation Error, n =" + str(N)))
        axs[1,0].legend()
        fig.show()  

    #splines
        axs[0,1].set_ylim([-1,2])
        axs[0,1].plot(xeval,fex,'ro-')
        axs[0,1].plot(xeval,yevalN,'g--',label='Natural') 
        axs[0,1].plot(xeval,yevalC,'y.--',label='Clamped')
        axs[0,1].set_title(("Splines, n =" + str(N)))
        axs[0,1].legend()
        # axs[0,1].semilogy()
             
        errN = abs(yevalN-fex)
        errC = abs(yevalC-fex)
        
        axs[1,1].semilogy(xeval,errN,'g--',label='Natural Error')
        axs[1,1].semilogy(xeval,errC,'y.--',label='Clamped Error')
        axs[1,1].set_title(str("Spline Error, n =" + str(N)))
        axs[1,1].legend()
        # fig.suptitle("Interpolation Equispaced")
        # fig.show()  
        #%% Boogying with the Chebyshev Nodes
        xint = np.zeros(N+1)
        for j in range(1,N+2):
           xint[j-1] = np.cos(np.pi*(2*j-1)/(2*(N+1)))
        # scale  for the interval
        m = (b-a)/2 
        c = (a+b)/2
        xint = m*xint+c
        xint = xint[::-1]
        ''' create interpolation data'''
        yint = np.zeros(N+1)
        ypint = np.zeros(N+1)
        for jj in range(N+1):
            yint[jj] = f(xint[jj])
            ypint[jj] = fp(xint[jj])
        
        ''' create points for evaluating the Lagrange interpolating polynomial'''
        Neval = 1000
        xeval = np.linspace(a,b,Neval+1)
        yevalL = np.zeros(Neval+1)
        yevalH = np.zeros(Neval+1)
        for kk in range(Neval+1):
          yevalL[kk] = eval_lagrange(xeval[kk],xint,yint,N)
          yevalH[kk] = eval_hermite(xeval[kk],xint,yint,ypint,N)
          
        # create cubic splines
        
        yp_b = np.zeros(2)
        yp_b[0] = fp(xint[0])
        yp_b[1] = fp(xint[N])
        
        # Natural
        (Mn,Cn,Dn) = create_natural_spline(yint,xint,N)
        yevalN = eval_cubic_spline(xeval,Neval,xint,N,Mn,Cn,Dn)
        
        #Clamped
        (Mc,Cc,Dc) = create_clamped_spline(yint,yp_b,xint,N)
        yevalC = eval_cubic_spline(xeval,Neval,xint,N,Mc,Cc,Dc)
        
    
        ''' create vector with exact values'''
        fex = np.zeros(Neval+1)
        for kk in range(Neval+1):
            fex[kk] = f(xeval[kk])
        
        # fig,axs = plt.subplots(2,2,figsize=(15,10),sharey="row")
        axs[2,0].set_ylim([-1,2])
        axs[2,0].plot(xeval,fex,'ro-')
        axs[2,0].plot(xeval,yevalL,'bs--',label='Lagrange') 
        axs[2,0].plot(xeval,yevalH,'c.--',label='Hermite')
        axs[2,0].set_title(str("Chebyshev Lagrange Polynomial, n =" + str(N)))
        axs[2,0].legend()
        # axs[2,0].semilogy()
        # plt.show()
             
        errL = abs(yevalL-fex)
        errH = abs(yevalH-fex)
        axs[3,0].semilogy(xeval,errL,'bs--',label='Lagrange Error')
        axs[3,0].semilogy(xeval,errH,'c.--',label='Hermite Error')
        axs[3,0].set_title(str("Chebyshev Approximation Error, n =" + str(N)))
        axs[3,0].legend()
        fig.show()  

    #splines
        axs[2,1].set_ylim([-1,2])
        axs[2,1].plot(xeval,fex,'ro-')
        axs[2,1].plot(xeval,yevalN,'g--',label='Natural') 
        axs[2,1].plot(xeval,yevalC,'y.--',label='Clamped')
        axs[2,1].set_title(("Chebyshev Splines, n =" + str(N)))
        axs[2,1].legend()
        # axs[2,1].semilogy()
             
        errN = abs(yevalN-fex)
        errC = abs(yevalC-fex)
        
        axs[3,1].semilogy(xeval,errN,'g--',label='Natural Error')
        axs[3,1].semilogy(xeval,errC,'y.--',label='Clamped Error')
        axs[3,1].set_title(str("Chebyshev Spline Error, n =" + str(N)))
        axs[3,1].legend()
        
        # fig.suptitle("Interpolation Chebyshev Nodes")
        
        fig.show()  
        
        #%% 
    
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
       

  # run the drivers only if this is called from the command line
driverN()       
driverS()