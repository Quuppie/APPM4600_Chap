"""
 This script illustrates the preformance and challenges for 
 the different interpolation techniques.  It also looks at plots
 of the different basis functions to help visually explain
 stability of the different methods. 
 This code is more for presentation in class than students to use
 independently.  I comment out the parts that I do not want to use 
 on a given lecture day. 
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

def driver():
    '''
    Call each of the included drivers in sequence.
    '''
    driver_monomail_basis_demo()
    driver_lagrange_basis_demo()
    # driver_sin_approx_demo()
    # driver_lagrange_vs_hermite()
    plt.show()
    
    
def driver_monomail_basis_demo():
    '''
    Plots the monomial basis functions.
    '''
    zs = np.linspace(-1, 1, 201)
    plt.figure(1, figsize=(10,6))
    for phi in monomial_basis(4):
        plt.plot(zs, phi(zs), '-')
    plt.title('monmial basis')
    plt.show()
def driver_lagrange_basis_demo():
    '''
    Plots some lagrange basis functions on some unequally spaced points.
    '''
    xs = np.linspace(0,1,5) #np.array([-1, -.5, 0, .25, 1])
    zs = np.linspace(-1, 1, 201)
    plt.figure(2, figsize=(10,6))
    plt.plot(zs, zs*0, 'k:')
    plt.plot(zs, zs*0+1, 'k:')
    for phi in lagrange_basis(xs):
        plt.plot(zs, phi(zs), '-')
        plt.plot(xs, phi(xs), 'k.')
    plt.title('A Lagrange basis on unequally spaced points')
    plt.show()
    
def driver_sin_approx_demo():
    '''
    Demonstrating approximation via interpolation
    '''
    f = lambda x: np.sin(x)
    n = 5 # number of sample points to use
    # choose a basis - comment out the other
    basis = monomial_basis(n)
    xs = np.linspace(-np.pi, np.pi, n)
    ys = f(xs)
    coeffs = get_interpolation_coefficients(xs, ys, basis)
    polynomial_text = ' + '.join([f'{c:.2f}x^{i}' for i, c in enumerate(coeffs)])
    # test the function on a finer grid
    zs = np.linspace(xs[0], xs[-1], 201)
    zs_eval = interp_eval(zs, coeffs, basis)
    plt.figure(3, figsize=(16,6))
    plt.plot(zs, f(zs), label='True function')
    plt.plot(zs, zs_eval, label='Interpolating Polynomial')
    plt.plot(xs, f(xs), 'k.', label='sample points')
    plt.text(-1, -1, '$' + polynomial_text + '$')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.title('Polynomial approximation by interpolation')
    #plt.show()
    
def driver_lagrange_vs_hermite():
    '''
    Compare Lagrange vs Hermite interpolation for a pair of functions
    '''
    function_derivative_pairs = [
        [lambda x: np.exp(x)*np.sin(x), lambda x: np.exp(x)*np.cos(x)
+np.exp(x)*np.sin(x), "e^x sin(x)"],
        [lambda x: np.exp(-16*x**2), lambda x: -32*x*np.exp(-16*x**2),"exp(-16x^2)"    ]
        ]
    # use one of these
    # f, df = function_derivative_pairs[0]
    for f, df, latex in function_derivative_pairs:
        zs = np.linspace(-1, 1, 101)
        exact = f(zs)
        #Ns = list(range(2, 21, 2)) #2, 4, ..., 20
        Ns = list(range(2, 21, 4))
        fig, axes = plt.subplots(len(Ns), 2, figsize=(20, 30), sharex=True)
        for i, N in enumerate(Ns):
            xs = np.linspace(-1, 1, N)
            ys = f(xs)
            ds = df(xs)
            basis = lagrange_basis(xs)
            lagrange_coeffs = get_interpolation_coefficients(xs, ys, basis)
            lagrange_eval = interp_eval(zs, lagrange_coeffs, basis)
            hermite_eval = hermite_interp_eval(zs, xs, ys, ds)
            axes[i][0].plot(zs, exact, 'k-', label='Exact')
            axes[i][0].plot(zs, lagrange_eval, 'g--', label='Lagrange')
            axes[i][0].plot(zs, hermite_eval, 'b--', label='Hermite')
            axes[i][1].semilogy(zs, np.abs(exact - lagrange_eval), 'g--', 
label='Lagrange')
            axes[i][1].semilogy(zs, np.abs(exact - hermite_eval), 'b--', 
label='Hermite')
            axes[i][0].set_ylabel(f'N={N}')
            axes[i][1].set_ylim(1e-16, 1e0)
        axes[0][0].set_title(f'Approximation of {latex}')
        axes[0][1].set_title('Error')
        axes[0][0].legend()
        #plt.show()
def monomial_basis(n):
    return [lambda x, i=i: x**i for i in range(n)]
def lagrange_polynomial(x, x_i, other_points):
    # f_i(x_j) = 0 for i =/= j
    # f_j(x_j) = 1
    # phi(x) = product of (x - x_j)/(x_i - x_j)
    y = 1 # start with 1 when making a product in a loop
    for x_j in other_points:
        y *= (x - x_j)/(x_i - x_j)
    return y
def remove_point_from_list(x_i, points):
    return [x_j for x_j in points if x_j != x_i]
def lagrange_basis(points):
    return [lambda x, x_i=x_i, other_points=remove_point_from_list(x_i, points): 
                lagrange_polynomial(x, x_i, other_points)
            for x_i in points]
    
def get_interpolation_coefficients(xs, ys, basis):
    M = np.array([[phi(x) for phi in basis] for x in xs])
    return la.solve(M, ys)
def interp_eval(zs, coeffs, basis):
    return sum(c*phi(zs) for c, phi in zip(coeffs, basis))
####################################
#
# Hermite Codes
#
####################################
from math import factorial
def hermite_interp_eval(zs, xs, ys, ds):
    '''
    Interpolate the values ys and the derivative values ds at
    the domain values xs then return the evaluation of the 
    interpolant at the values zs.
    '''
    assert len(xs) == len(ys)
    assert len(xs) == len(ds)
    repeated_values = np.repeat(xs, 2)
    inverleaved_values = np.array(list(zip(ys, ds))).flatten()
    # get monomial coefficients from Newton divided difference table
    coeffs = newton_interp(repeated_values, inverleaved_values)
    return newton_eval(zs, coeffs)
def newton_interp(xs, fs):
    '''
    Interpolate the values in the array fs at the points xs. If xs are
    repeated, the repeated values must be adjacent. It is easiest to provide
    them in increasing order. If xs are repeated, then the provided f values
    are to be interpreted as derivatives.
    '''
    assert len(xs) == len(fs)
    n = len(xs)
    # check derivative value
    order = [0]
    for i in range(1,n):
        if xs[i-1] == xs[i]:
            order += [order[i-1]+1]
        else:
            order += [0]
    ds = [fs[0]]
    for i in range(1,n):
        ds += [fs[i-order[i]]]
    # divided difference table
    for col in range(1, n):
        for row in range(n-1, col-1, -1):
            if order[row] >= col:
                ds[row] = fs[row-(order[row]-col)]/factorial(col)
            else:
                ds[row] = (ds[row] - ds[row-1])/(xs[row] - xs[row-col])
    # Horner's Rule
    for i in range(len(ds)-1-1,-1,-1):
        for j in range(i,len(ds)-1):
            ds[j] -= xs[i]*ds[j+1]
            
    return ds
def newton_eval(zs, coeffs):
    '''
    Evaluate a polynomial given coefficients from newton_interp
    '''
    ret = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        ret = ret*zs + c
    return ret
if __name__ == '__main__':
    # run the drivers only if this is called from the command line
    driver()
    
    
