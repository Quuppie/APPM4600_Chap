#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 10:22:00 2026

@author: Cayden Stratford
"""

import numpy as np
import matplotlib.pyplot as plt

#%%Define a function
def pexpanded(x):
    pexp = pow(x,9) - 18*pow(x,8) + 144*pow(x,7) - 672*pow(x,6) + 2016*pow(x,5) - 4032*pow(x,4)+ 5376*pow(x,3) - 4608*pow(x,2) + 2304*x - 512
    return pexp

def psimple(x):
    pwee = pow((x-2),9)
    return pwee

#%%Main

x = np.arange(1.920,2.080,0.001)
pexp = np.zeros(len(x))
psimp = np.zeros(len(x))

for i in range(0,len(x)):
    pexp[i] = pexpanded(x[i])
    psimp[i] = psimple(x[i])
    
fig, (ax1,ax2) = plt.subplots(2,1,sharex="col")
plt.subplot(2,1,1)
plt.plot(x,psimp,'blue')
plt.subplot(2,1,2)
plt.plot(x,pexp,'orange')

plt.suptitle("Coherent vs Expanded: (x-2)^9")
plt.savefig("xto9",dpi=300)
plt.show()

#%% Problem 5 functions

def nosubtrac(x,delta):
    trigout = -2*np.sin((2*x+delta)/2)*np.cos(delta/2)
    return trigout
def subtrac(x,delta):
    out = np.cos(x+delta) - np.cos(x)
    return out
def funkyalgorithm(x,delta):
    bruh = -delta*np.sin(x) - pow(delta,2)/2*np.cos(x+delta)
    return bruh

#%% Main 2
xray = np.array([np.pi, pow(10,5)])
delta = np.logspace(-16,0,17)
scrung = np.zeros(len(delta))
bung = np.zeros(len(delta))
krung = np.zeros(len(delta))
fig, ax = plt.subplots(2,1,sharex="all")
for i in (0,1):
    for j in np.arange(0,16):
        scrung[j] = nosubtrac(xray[i],delta[j])
        bung[j] = subtrac(xray[i],delta[j])
    plt.subplot(2,1,(i+1))
    plt.plot(delta,abs(scrung-bung))
    plt.xscale('log')
    # plt.plot(delta,bung)
    plt.yscale('log')
    plt.grid('log')
    plt.xlabel('log(delta)')
    plt.ylabel('log(epsilon)')
    plt.suptitle("Trig-Based Algorithm")
plt.savefig("trig",dpi=300)
plt.show()

for i in (0,1):
    for j in np.arange(0,16):
        krung[j] = funkyalgorithm(xray[i],delta[j])
    plt.subplot(2,1,(i+1))
    plt.plot(delta,abs(krung))
    plt.xscale('log')
    # plt.plot(delta,bung)
    plt.yscale('log')
    plt.grid('log')
    plt.xlabel('log(delta)')
    plt.ylabel('log(epsilon)')
    plt.suptitle("Personal Algorithm")
    
plt.savefig("personalalgorithm",dpi=300)
plt.show()


        