
"""
 This script tests the convergence of gauss-legendre quadrature.
 Different functions are available  to explore the performance.
 The code imports gauss_legendre.py.
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


# get the lgwts routine and numpy from gauss_legendre.py
from gauss_legendre import *
# get plot routines
import matplotlib.pyplot as plt

# interval of integration [a,b]
a = -1.; b = 2.
# make some test functions to integrate and truth values of integral
# TRYME: comment and uncomment to try them!
#        make sure to adjust I_true values if using different interval!
#f = lambda x: x**3; I_true = 3.75; labl = '$x^3$'
#f = lambda x: x**5; I_true = 10.5; labl = '$x^5$'
#f = lambda x: x**11; I_true = 1365./4.; labl = '$x^{11}$'
f = lambda x: np.exp(np.cos(x)); I_true = 5.7959297382322493; labl = '$\exp(\cos(x))$'

# machine eps in numpy
eps = np.finfo(float).eps
# number of nodes/weights per gaussian quad rule
ns = np.arange(1,15); nN = len(ns)

# loop over orders (ns) and compute approximate integral + error
I_hats = np.zeros((nN,))
errors = np.zeros((nN,))
for iN  in range(nN):
  N = ns[iN]
  [x,w] = lgwt(N,a,b)
  I_hat = np.sum(f(x)*w)
  
  I_hats[iN] = I_hat
  errors[iN] = np.abs(I_hat-I_true)/I_true
  # adjust zero errors for log plot
  if errors[iN] < eps:
    errors[iN] = eps

print('w=',w)

plt.semilogy(ns,errors,'o--',label=labl)
plt.xticks(ns)
plt.legend()
plt.title('Testing Gauss-Legendre Quadrature')
plt.ylabel('Relative Error')
plt.xlabel('N - # points and # weights - order 2N-1')
plt.show()
