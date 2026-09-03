#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 10:22:00 2026

@author: Cayden Stratford
"""

import numpy as np
import matplotlib.pyplot as plt

#Define a function
def pexpanded(x):
    pexp = pow(x,9) - 18*pow(x,8) + 144*pow(x,7) - 672*pow(x,6) + 2016*pow(x,5) - 4032*pow(x,4)+ 5376*pow(x,3) - 4608*pow(x,2) + 2304*x - 512
    return pexp

def psimple(x):
    pwee = pow((x-2),9)
    return pwee

x = np.arange(1.920,2.080,0.001)
pexp = np.zeros(len(x))
psimp = np.zeros(len(x))

for i in [0,len(x)]:
    pexp[i] = pexpanded(x[i])
    psimp[i] = psimple(x[i])
    
    