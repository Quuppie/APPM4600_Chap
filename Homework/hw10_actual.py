"""
 This script tests the use of different composite quadrature methods
 for approximating integrals.  There is an option to adjust the commented
 lines to illustrate convergence when the number of panels is increased.
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

def driver():
    
    f = lambda x: np.e**(x) * np.cos(x)
    a = 0
    b = np.pi
    
    # exact integral
    I_ex = -12.070346316389632
    
    N = 301
# for simpson's n must be even.
# n+1 = number of pts.
    ntest = np.arange(2,N+1,step=2)

    errorT = np.zeros(len(ntest))
    errorS = np.zeros(len(ntest))

    for j in range(0,len(ntest)):
        n = ntest[j]

        I_trap = CompTrap(a,b,n,f)
        errorT[j] = abs(I_ex-I_trap)

        I_simp = CompSimp(a,b,n,f)
        errorS[j] = abs(I_ex-I_simp)

    print('n = ', ntest[-1])
    print('trapezoidal absolute error = ', errorT[-1])
    print('Simpson absolute error = ', errorS[-1])

    errorT = abs(errorT / I_ex)
    errorS = abs(errorS / I_ex)
    
    print('n = ', ntest[-1])
    print('trapezoidal relative error = ', errorT[-1])
    print('Simpson relative error = ', errorS[-1])
    
    # observed order of convergence from the last two runs
    rateT = np.log(errorT[-1]/errorT[-2])/np.log(ntest[-2]/ntest[-1])
    rateS = np.log(errorS[-1]/errorS[-2])/np.log(ntest[-2]/ntest[-1])
    print('observed order (trap) = ', rateT)
    print('observed order (Simpson) = ', rateS)
    
    
    # reference slopes O(h^2) and O(h^4), with h = (b-a)/n
    h = (b-a)/ntest
    plt.loglog(ntest,errorT,'o-',label='Composite trapezoidal')
    plt.loglog(ntest,errorS,'s-',label='Composite Simpson')
    plt.loglog(ntest,errorT[0]*(h/h[0])**2,'k--',label='$O(h^2)$')
    plt.loglog(ntest,errorS[0]*(h/h[0])**4,'k:',label='$O(h^4)$')
    plt.xlabel('n (number of subintervals)')
    plt.ylabel('relative error')
    plt.title("Error vs Intervals")
    plt.legend()
    plt.show()

        
def CompTrap(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    
    I_trap = h*f(xnode[0])*1/2
    
    for j in range(1,n):
         I_trap = I_trap+h*f(xnode[j])
    I_trap= I_trap + 1/2*h*f(xnode[n])
    
    return I_trap     

def CompSimp(a,b,n,f):
    h = (b-a)/n
    xnode = a+np.arange(0,n+1)*h
    I_simp = f(xnode[0])

    nhalf = n/2
    for j in range(1,int(nhalf)+1):
         # odd part
         I_simp = I_simp +4*f(xnode[2*j-1])
         # even part (interior only; x_n is added below)
         if j < nhalf:
             I_simp = I_simp+2*f(xnode[2*j])
    I_simp= I_simp + f(xnode[n])
    
    I_simp = h/3*I_simp
    
    return I_simp     

    
driver()    
