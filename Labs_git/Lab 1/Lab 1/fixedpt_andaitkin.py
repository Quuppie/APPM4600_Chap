"""
 This script explores the use of the fixed point method.  
 Two functions are considered that have different properties.
 I like to use this code before I talk about convergence analysis
 for the fixed point method as motivation.
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



# import libraries
import numpy as np
    
def driver():


# test functions 
    f1 = lambda x: 1+0.5*np.sin(x)
# fixed point is alpha1 = 1.4987....

    f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09... 

    Nmax = 100
    tol = 1e-6

# test f1 '''
    x0 = 0.0
    [xstar1,ier] = fixedpt(f1,x0,tol,Nmax)
    print('the approximate fixed point is:',xstar1)
    print('f1(xstar):',f1(xstar1))
    print('Error message reads:',ier)
    
# #test f2 '''
#      x0 = 0.0
#      [xstar2,ier] = fixedpt(f2,x0,tol,Nmax)
#      print('the approximate fixed point is:',xstar2)
#      print('f2(xstar):',f2(xstar2))
#      print('Error message reads:',ier)
#Make Aitken's approx
    x0 = 0.0
    [xaitken,ier] = aitkenmoment(xstar1,tol,Nmax)
    
    
     
    return [xstar1, xaitken]

# define routines
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
    
def aitkenmoment(pseq,tol,Nmax):
    count = 0
    paitk = np.zeros((Nmax,1))
    while (count<Nmax-2):
        count = count+1
        paitk[count] = pseq[count]-((pseq[count+1]-pseq[count])**2)/(pseq[count+2]-2*pseq[count+1]+pseq[count])    #defining Aitkin's
        if ((abs(paitk[count]-paitk[count-1]))<tol):
            ier = 0
            return[paitk,ier]
    ier=1
    return [paitk,ier]
    

[xstar1, xaitken] = driver()