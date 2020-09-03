# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 12:18:32 2020

@author: 1026691
"""
"""Measurement update function practice of localization"""


p=[0.2, 0.2, 0.2, 0.2, 0.2]
world=['green', 'red', 'red', 'green', 'green']
Z = 'red'
pHit = 0.6
pMiss = 0.2


def sense(p,Z):
    q=[]
    
    for i in range(len(p)):
        hit = (Z==world[i])
        q.append(p[i] * (hit*pHit + (1 - hit)* pMiss))
        return q
print(sense(p,Z))