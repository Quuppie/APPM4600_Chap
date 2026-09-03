#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:26:23 2026

@author: groot
"""

import matplotlib.pyplot as plt
import numpy as np
## Practice
X = np.linspace(0,2*np.pi,100) ; Ya = np.sin(X) ; Yb = np.cos(X)

plt.plot(X,Ya)
plt.plot(X,Yb)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
#%%
x = np.linspace(0,123,42)
y = np.arange(0,126,3)

print("Length of x is",len(x), "and length of y is",len(y))

firstx = x[0:3]
firsty = y[0:3]

print("The first three entries of x are",firstx)

w = 10**(-np.linspace(1,10,10))
x = np.arange(0,len(w),1)
s = 3*w
plt.figure()
#fig,ax = plt.subplots()
plt.grid('minor')
plt.semilogy(x,w,label="w")
plt.semilogy(x,s,label="s")
plt.xlabel("x")
plt.ylabel("log(y)")
plt.title("Exponentials")

plt.legend()
plt.show()

plt.savefig("logs.jpg")