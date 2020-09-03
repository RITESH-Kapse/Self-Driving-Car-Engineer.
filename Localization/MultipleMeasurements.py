# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 12:46:29 2020

@author: 1026691
"""

'''Multiple measurements '''

p=[0.2, 0.2, 0.2, 0.2, 0.2]
world=['green', 'red', 'red', 'green', 'green']
measurements = ['red','green']
pHit = 0.6
pMiss = 0.2


def sense(p,Z):
    q=[]
    
    for i in range(len(p)):
        hit = (Z==world[i])
        q.append(p[i] * (hit*pHit + (1 - hit)* pMiss))
    
    s = sum(q)
    
    for i in range(len(p)):
        q[i]=q[i]/s
    return q


for k in range(len(measurements)):
    p = sense(p,measurements[k])    
    
print(p)