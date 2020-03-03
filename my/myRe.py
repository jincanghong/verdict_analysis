#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re

def del_repeat(txt,repeat=""):
    '''
    将连续的重复字符替换成一个字符。
    
    txt纯文本。
    repeat：需要去重的字符。
    '''
    if repeat:
        return re.sub(repeat+"+",repeat,txt)
    else:
        return txt

def match_provice(token):
    '''
    匹配当前词汇是否为省份
    '''
    province=["北京市","天津市","上海市","重庆市","河北省","山西省","辽宁省","吉林省","黑龙江省",
                "江苏省","浙江省","安徽省","福建省","江西省","山东省","河南省","湖北省","湖南省",
                "广东省","海南省","四川省","贵州省","云南省","陕西省","甘肃省","青海省","台湾省",
                "内蒙古自治区","广西壮族自治区","西藏自治区","宁夏回族自治区","新疆维吾尔自治区",
                "香港特别行政区","澳门特别行政区","北京","天津","上海","重庆","河北","山西","辽宁",
                "吉林","黑龙江","江苏","浙江","安徽","福建","江西","山东","河南","湖北","湖南","广东",
                "海南","四川","贵州","云南","陕西","甘肃","青海","台湾","内蒙古","广西","西藏","宁夏",
                "新疆","香港","澳门"]
    province=set(province)
    if token in province:
        return True
    else:
        return False

def filter_time(txt,year=True,month=True,day=True):
    assert isinstance(txt,str)
    return re.sub("[一二三四五六七八九十零〇0-9]+[年月日]","",txt)

def filter_location(tokens,others=[]):
    '''
    tokens:词汇列表。
    others表示需要补充过滤的词汇。
    
    注意！还有问题，为什么单独的街道可以匹配？？？？？？？？？？？？？？？？？？
    '''
    assert isinstance(tokens,list)
    res=[]
    for token in tokens:
        if re.match("^.+?[(街道)县市区镇村路]$",token) is not None:
            #print("match:",re.match("^.+?[(街道)县市区]$",token))
            continue
        elif match_provice(token):
            continue
        elif token in set(others):
            continue
        else:
            res.append(token)
    return res


if __name__=="__main__":
    # txt="三十个月二〇〇八年-208年1年三年一次二十二次"
    # a=tmp(txt)
    # print(a)
    #print(del_repeat(",,,,,abaasdfjklskdjfabbbbc,",repeat="a"))
    #a="jieba.lcut('1994年 -11 十二月 ')"
    #a=filter_time(a)
    #print(a)
    #a=["六十二","三"]
    #a=filter_number(a)
    #a=filter_location(a)
    #print(a)
    pass