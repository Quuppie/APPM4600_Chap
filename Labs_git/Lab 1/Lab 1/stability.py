"""
 This program is meant to illsturate the hidden dangers of 
 floating point operations.
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



########################################################################
# This python script presents examples of algorithm stability illustrated
# on the notes (equivalent to stability_example.m)
# APPM 4600 Fall 2022
########################################################################
import matplotlib.pyplot as plt
import numpy as np; # import numpy

# number of terms 
N = 40
#  example 1
# recursion vs formula
p = np.zeros(N) # actual sequence
p1 = np.zeros(N) # approximation
p2 = np.zeros(N) # recursion relation

epsilon = 0.125*1.0e-8
# Initialize the approximations
p[0] = 1.
p[1] = 1./3.
p1[0] = 1.
p1[1] = 1./3.
p2[0] = 1.
p2[1] = 1./3.

# we compute each sequence as described in the notes.
for j in range(2,N):
    p[j] = np.power(1./3.,j)
    p1[j] = np.power(1./3.,j) - epsilon*np.power(3.,j)
    p2[j] = (10./3.)*p2[j-1]-p2[j-2]
  
  
# plot curves for the first N iterates for the actual sequence, approximation and
# recursion relation
x = np.arange(0,N)
plt.plot(x[2:N],p[2:N],'r-')
plt.xlabel('iteration')
plt.ylabel('value')
plt.plot(x[2:N],p1[2:N],'b-')
plt.plot(x[2:N],p2[2:N],'g-')
plt.show()

input()
# plot of the absolute errors between the sequence and both methods
plt.semilogy(x[2:N],np.abs(p[2:N]-p1[2:N]),'r-')
plt.xlabel('iteration')
plt.ylabel('absolute err')
plt.semilogy(x[2:N],np.abs(p[2:N]-p2[2:N]),'b-')
plt.show()
input()
## plot of relative errors between the sequence and both methods
plt.semilogy(x[2:N],np.abs(p[2:N]-p1[2:N])/np.abs(p[2:N]),'r-')
plt.semilogy(x[2:N],np.abs(p[2:N]-p2[2:N])/np.abs(p[2:N]),'b-')
plt.xlabel('iteration')
plt.ylabel('relative err')
plt.show()

