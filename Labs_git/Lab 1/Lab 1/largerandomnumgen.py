#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 14:03:16 2026

@author: groot
"""
import matplotlib.pyplot as plt

import numpy as np
 
N = 10 #steps

p = np.zeros(N)
p1 = np.zeros(N)
p2 = np.zeros(N)
epsilon = 0.125e10

for j in [2,N]:
    p[j] = 1*(1/3)^float(j)
    p1[j] = 1*(1/3)^float(j) + epsilon*(3)^float(j)
    p2[j] = 10/3*p2(float(j)-1)

