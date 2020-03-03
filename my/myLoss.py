#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def mse(logits,labels):
    '''
    logits,labels都是一维numpy数组。
    rmse等于logits和labels求差值，再平方，所以结果的logits和labes的平方相加取均值。
    '''
    return np.sum(np.square(logits-labels))/len(logits)

def rmse(logits,labels):
    '''
    logits,labels都是一维numpy数组。
    rmse等于logits和labels求差值，再平方，所以结果的logits和labes的平方相加取均值再开根号。
    即mse加上一个根号。
    '''
    return np.sqrt(mse(logits=logits,labels=labels))

def var(labels):
    '''
    labels都是一维numpy数组。对数组里的值求方差。
    '''
    aver=np.mean(labels)
    return np.mean(np.square(labels-aver))

def r_squared(mse,var):
    assert var>=1e-12
    return 1-mse/var

def F1():
    raise NotImplementedError()

def acc():
    raise NotImplementedError()

def rec():
    raise NotImplementedError()



if __name__=="__main__":
    a=np.array([0,4,10])
    b=np.array([0,0,9])
    m=mse(a,b)
    rm=rmse(a,b)
    v=var(b)
    rs=r_squared(m,v)
    print(m,rm,v,rs)
    pass