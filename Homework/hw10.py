"""
 This script tests Romberg integration for approximating integrals.
 The first column of the Romberg table is built with the composite
 trapezoidal rule and the remaining columns come from Richardson
 extrapolation.  The error in the diagonal entries is plotted as the
 number of panels is increased.
"""

#############################################
"""
Copyright (C) 2026  Adrianna M. Gillman

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


def driver():

    f = lambda x: np.e**(x)*np.cos(x)
    a = 0
    b = np.pi

    # exact integral
    I_ex = -12.070346316389632

    nrefine = 10

# number of panels: 1, 2, 4, ..., 2^nrefine
    ntest = 2**np.arange(0,nrefine+1)
    trapzez = np.zeros(len(ntest))
    errt = np.zeros(len(ntest))
    trapzsim = np.zeros(len(ntest))
    errs = np.zeros(len(ntest))
    for j in range(0,len(ntest)):
        trapzez[j] = CompTrap(a,b,ntest[j],f)
        errt[j] = abs(I_ex-trapzez[j])
        
    for j in range(0,len(ntest)):
        trapzsim[j] = SimpsonsTrap(a,b,ntest[j],f)
        errs[j] = abs(I_ex-trapzsim[j])
    
    
    print('Error of trapezoidal approximation at each order = ', errt)
    plt.loglog(ntest,errt,'bo-')
    plt.loglog(ntest,errs,'ro-')
    plt.xlabel('n')
    plt.ylabel('absolute error')
    plt.title("Error vs Iteration Trapezoidal")
    plt.show()
    
    plt.figure()
    plt.plot(ntest,trapzsim)

#     Rom = rombergTrap(a,b,nrefine,f)
#     print("Rom is", Rom)
# # column j is only filled from row j down
#     for j in range(0,nrefine+1):
#         print('absolute error in column', j, '= ', abs(I_ex-Rom[j:,j]))

#     err = abs(I_ex-np.diag(Rom))

#     print('absolute error of best approximation at each order = ', err)

#     plt.semilogy(ntest,err,'o-')
#     plt.xlabel('n')
#     plt.ylabel('absolute error')
#     plt.title("Error vs Iteration Trapezoidal")
#     plt.show()


def rombergTrap(a,b,nrefine,f):
# number of panels: 1, 2, 4, ..., 2^nrefine
    ntest = 2**np.arange(0,nrefine+1)

    nn = len(ntest)

    Rom = np.zeros((nn,nn))

# first column: composite trapezoidal rule with n panels
    for j in range(0,nn):
        Rom[j,0] = CompTrap(a,b,ntest[j],f)

# remaining columns: Richardson extrapolation
    for j in range(1,nn):
        for i in range(j,nn):
            Rom[i,j] = Rom[i,j-1] + (Rom[i,j-1]-Rom[i-1,j-1])/(4**j-1)

    return Rom


def CompTrap(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h

    I_trap = h*f(xnode[0])*1/2

    for j in range(1,n):
         I_trap = I_trap+h*f(xnode[j])
    I_trap= I_trap + 1/2*h*f(xnode[n])

    return I_trap

def SimpsonsTrap(a,b,n,f):  #Reference page 206, ch. 4.4. Still working on this
    h = (b-a)/n
    XI0 = f(a)+f(b)
    XI1 = 0
    XI2 = 0
    for i in range(1,n-1):
        X = a + i*h
        if X % 2 == 0:
            XI2 = XI2+f(X)
        else:
            XI1 = XI1+f(X)
# %%
    XI = h/3*(XI0+4*XI1+2*XI2)

    return(XI)
        


driver()
