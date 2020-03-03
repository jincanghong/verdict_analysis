#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

INF=1e12

def cos(a,b):
    '''两个相同形状的数组进行numpy运算。'''
    raise NotImplemented()

def cos1d(a,b):
    '''
    a和b都是一维numpy数组。
    '''
    dot=np.sum(a*b)
    abs_a=np.sum(np.square(a))
    abs_b=np.sum(np.square(b))
    res=np.sqrt(abs_a*abs_b)
    if res<1e-12:
        res=INF
    else:
        res=dot/res
    return res

if __name__=="__main__":
    a=np.array([[[2,3,4]]])
    b=np.array([[[2,3,0]]])
    print(cos1d(a,b))
    pass