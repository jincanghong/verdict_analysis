#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import copy
import numpy as np
from collections import OrderedDict

def new_dir(dir1,dir2):
    '''将dir2文件夹建立在dir1中'''
    assert os.path.exists(dir1)
    tmp=os.path.join(dir1,dir2)
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    return

def file_readlines(path):
    with open(path,"r",encoding="utf-8") as f:
        txt=f.readlines()
    return txt

def addv_dict(a,b):
    '''
    将字典中键相同的值相加。
    '''
    # 对短的字典进行遍历
    if len(a)<len(b):
        aa=copy.deepcopy(b)
        bb=a
    else:
        aa=copy.deepcopy(a)
        bb=b
    for bk,bv in bb.items():
        if bk in aa.keys():
            aa[bk]+=bv
        else:
            aa[bk]=bv
    return aa

def sortk_dict(d):
    l=[(k,v) for k,v in d.items()]
    l.sort(key=lambda x:x[0])
    return OrderedDict(l)

def sortv_dict(d):
    l=[(k,v) for k,v in d.items()]
    l.sort(key=lambda x:x[1])
    return OrderedDict(l)

def copy_file(src,dest):
    if os.name=="nt":
        os.system("copy %s %s"%(src,dest))
    elif os.name=="posix":
        # 这个还没测试
        os.system("cp %s %s"%(src,dest))
    else:
        print("os.name should be 'nt' or 'posix'.")
    return

def format_question(s):
    # input a string, output a formated string.
    # Reserve one '_' and chinese, english, and number. Remove punctuation.
    #s=re.sub("[<:\s+\-={}`\.\!\/,$%^;*(+\"\')\]\\\\+>|\[+()?【】“”！，。？；、▪《》~@·☆‘’：#￥%……&*（）]+","",s)
    s=re.sub("[^\u4e00-\u9fd0^\d^a-z^A-Z^_]","",s)
    s=re.sub('_+',"_",s)
    return s

def get_right_name(s):
    # 有些实体的第一个值就是括号，比如['22692400', '（真本）千金方']，所以如果取括号前的内容就会空值，然后overlap就会报错，所以还是用re把括号内内容替换成空比较好。
    return re.sub("（.*?）","",s)
    ## if containing "（", get the name before "（".
    #return s.split("（")[0]
            
def json_file_to_dict(path):
    with open(path,"r",encoding="utf-8") as f:
        d=json.load(f)
    return d

def write_as_json_file(d,path):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(d,f,ensure_ascii=False)
        
def filter_except_blank_space(s):
    # filter punctuation. Reserve chinese, english, number and blank.
    return re.sub("[^\u4e00-\u9fd0^\d^a-z^A-Z^ ]","",s)
    #return re.sub("[_<:+\-={}`\.\!\/,$%^;*(+\"\')\]\\\\+>|\[+()?【】“”！，。？；、▪《》~@·☆‘’：#￥%……&*（）]+","",s)

def filter_all(s):
    # filter punctuation. Reserve chinese, english, and number.
    return re.sub("[^\u4e00-\u9fd0^\d^a-z^A-Z]","",s)
    #return re.sub("[_<:\s+\-={}`\.\!\/,$%^;*(+\"\')\]\\\\+>|\[+()?【】“”！，。？；、▪《》~@·☆‘’：#￥%……&*（）]+","",s)

def write_text_into_file(text,path):
    with open(path,"w",encoding="utf-8") as f:
        f.write(text)

def get_text_from_file(path):
    with open(path,"r",encoding="utf-8") as f:
        res=f.read()
    return res
            
def reverse_dict(d):
    '''
    当值唯一的时候，才能把值和键颠倒
    '''
    res={}
    for k,v in d.items():
        res[v]=k
    return res

def get_dict_by_line_str2int(path,sep=" "):
    '''
    一行存词典的一项，一行内第一个值为key(str)，第二个值为value(int)，k和v之间用空格分隔。
    逐行读取文件，生成字典并返回字典。
    '''
    res={}
    with open(path,'r',encoding="utf-8") as f:
        for line in f.readlines():
            if line[-1]=="\n":
                line=line[:-1]
            line=line.split(sep)
            if len(line)==2:
                res[line[0]]=int(line[1])
            else:
                print("get_dict_by_line_str2int() 有问题！一行不为两个元素：",line)
    return res

def get_dict_by_line_int2str(path,sep=" "):
    '''
    一行存词典的一项，一行内第一个值为key(str)，第二个值为value(int)，k和v之间用空格分隔。
    逐行读取文件，生成字典并返回字典。
    '''
    res={}
    with open(path,'r',encoding="utf-8") as f:
        for line in f.readlines():
            if line[-1]=="\n":
                line=line[:-1]
            line=line.split(sep)
            if len(line)==2:
                res[int(line[0])]=line[1]
            else:
                print("get_dict_by_line_int2str() 有问题！一行不为两个元素：",line)
    return res

def get_dict_by_line_str2str(path,sep=" "):
    '''
    一行存词典的一项，一行内第一个值为key(str)，第二个值为value(int)，k和v之间用空格分隔。
    逐行读取文件，生成字典并返回字典。
    '''
    res={}
    with open(path,'r',encoding="utf-8") as f:
        for line in f.readlines():
            if line[-1]=="\n":
                line=line[:-1]
            line=line.split(sep)
            if len(line)==2:
                res[line[0]]=line[1]
            else:
                print("get_dict_by_line_str2str() 有问题！一行不为两个元素：",line)
    return res

def get_list_by_line(file_path):
    '''
    读取文件，形成列表。
    文件中一行对应列表中的一项，列表中每一项为一个字符串。
    '''
    with open(file_path,"r",encoding="utf-8") as f:
        d=[i for i in f.read().split("\n")]
    return d

def add_line_into_txt(line,path):
    with open(path,"a",encoding="utf-8") as f:
        f.write(line+"\n")

# find file
def find_file_path_1st(folder,filename):
    '''
    在某一文件夹下查找是否存在某个文件。
    存在则返回文件的路径，不存在则返回空字符串。
    找到第一个就返回结果。
    '''
    if os.path.exists(os.path.join(folder,filename)):
        return os.path.join(folder,filename)
    else:
        next_folders=[i for i in os.listdir(folder) if os.path.isdir(os.path.join(folder,i))]
        for next_folder in next_folders:
            res=find_file_path_1st(os.path.join(folder,next_folder),filename)
            if res!="":
                return res
        return ""

def find_file_path_all(folder,filename,res):
    '''
    在某一文件夹下查找是否存在某个文件。
    存在则返回文件的路径，不存在则返回空字符串。
    返回所有找到的文件的路径，存在res里
    '''
    if os.path.exists(os.path.join(folder,filename)):
        res.append(os.path.join(folder,filename))
    next_folders=[i for i in os.listdir(folder) if os.path.isdir(os.path.join(folder,i))]
    for next_folder in next_folders:
        find_file_path_all(os.path.join(folder,next_folder),filename,res)
    del next_folders

def overlap(a_list,b_list):
    '''
    计算两个列表的重复元素数。
    '''
    if len(b_list)==0:
        return 0
    cnt=0
    b2=copy.copy(b_list)
    #print(b_list)
    for i in a_list:
        if i in b2:
            cnt+=1
            b2.remove(i)
            #print("overlap:",i)
    return cnt

def delete_stop_words(words_list,stops_list):
    if stops_list:
        res=[]
        for word in words_list:
            if word not in stops_list:
                res.append(word)
        return res
    else:
        return words_list

def arg_max_numpy2d(numpy2d):
    max_1d=np.argmax(numpy2d,axis=1)
    maxv=0
    maxp=(0,max_1d[0])
    for i in range(len(numpy2d)):
        tmp=numpy2d[i][max_1d[i]]
        if maxv<tmp:
            maxv=tmp
            maxp=(i,max_1d[i])
    return maxp,maxv

# def read_dictionary(path):
#     # 格式：词汇 频率 词性
#     # 返回dict，格式：{word:(freq,pos)}
#     res={}
#     with open(path,'r',encoding="utf-8") as f:
#         line=line.strip().split(" ")
#         if len(line)==3:
#             res[line[0]]=(int(line[1]),line[2])
#         else:
#             print("get_dict_by_line_str2int() 有问题！一行不只三个元素（词汇 频率 词性）：",line)
#     return res

def write_dict_into_file_str2int(d,path,sep=" ",sort=False,reverse=False):
    '''
    把字典写入文件，每行内容：str \t int。
    sort=True，则按int对字典的每一项进行排序，否则不排序。
    reverse=True，按降序对词典进行排序，否则升序。
    '''
    if sort:
        d=[(k,v) for k,v in d.items()]
        d.sort(key=lambda x:x[1],reverse=reverse)
        with open(path,"w",encoding="utf-8") as f:
            for i in d:
                f.write(str(i[0])+"\t"+str(i[1])+"\n")
    else:
        with open(path,"w",encoding="utf-8") as f:
            for k,v in d.items():
                f.write(str(k)+"\t"+str(v)+"\n")
    return

def write_log(txt,log_path,note=""):
    with open(log_path,"a",encoding="utf-8") as f:
        if note:
            f.write(note+"\n")
        f.write(txt+"\n")
    return
    
if __name__=="__main__":
    #res={1:[1,2],-1:[2,3]}
    #print(sortk_dict(res)[-1])
    new_dir("./__pycache__/","./data")
    pass