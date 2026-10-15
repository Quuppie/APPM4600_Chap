#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:26:53 2026

@author: boot
"""

import numpy as np
import matplotlib.pyplot as plt

def lineMaker(x0,x1,y0,y1,alpha):
    
    falpha = (y1-y0)/(x1-x0)*(alpha-x0)+y0

    return (alpha,falpha)

#Code for a linear line
x0 = 1
x1 = 3

y0 = 5
y1 = -2

alpha = 2

(alpha, falpha) = lineMaker(x0,x1,y0,y1,alpha)


