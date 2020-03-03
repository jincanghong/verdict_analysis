#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import copy
import jieba
from collections import OrderedDict
from collections import Counter
import myUtils

def filter_freq(vocab_list,freq_dict,freq_lower_bound,freq_upper_bound,skip):
    '''
    skip：是否跳过不在频率字典中的词。
    '''
    if freq_upper_bound==0:
        return []
    if len(freq_dict)==0:
        if skip:
            return []
        else:
            return vocab_list
    res=[]
    for voc in vocab_list:
        if voc in freq_dict:
            if freq_dict[voc]<=freq_upper_bound and freq_dict[voc]>=freq_lower_bound:
                res.append(voc)
        elif skip is not True:
            res.append(voc)
    return res

def read_vocab(path):
    '''
    读取字典文件（utf-8格式），一行为一个词。
    返回集合。
    '''
    assert os.path.exists(path)
    with open(path,"r",encoding="utf") as f:
        lines=f.readlines()
    vocab=[]
    for line in lines:
        if line[-1]=="\n":
            vocab.append(line[:-1])
        elif line!="":
            vocab.append(line)
    return set(vocab)

def write_vocab(vocab_list,path):
    with open(path,"w",encoding="utf") as f:
        for vocab in vocab_list:
            f.write(vocab+"\n")  
    return

def file_freq(path):
    with open(path,"r",encoding="utf") as f:
        lines=f.readlines()
    vocab=[]
    for line in lines:
        vocab+=list(line)
        vocab+=jieba.lcut(line)
    return Counter(vocab)

def files_freq(dir):
    files=os.listdir(dir)
    vocab={}
    for file in files:
        file=os.path.join(dir,file)
        print(file)
        vocab=myUtils.addv_dict(vocab,file_freq(file))
    return vocab

def file_vocab(path):
    with open(path,"r",encoding="utf") as f:
        lines=f.readlines()
    vocab=[]
    for line in lines:
        vocab+=list(line)
        vocab+=jieba.lcut_for_search(line)
    return list(set(vocab))

def files_vocab(dir):
    '''
    返回列表。
    '''
    files=os.listdir(dir)
    vocab=[]
    for file in files:
        file=os.path.join(dir,file)
        print(file)
        vocab+=file_vocab(file)
    return list(set(vocab))

def del_stops(vocabs,stops):
    '''
    vocabs,stops均为列表，存有词汇。
    '''
    res=[]
    if stops==[]:
        res=vocabs
    else:
        for vocab in vocabs:
            if vocab not in stops:
                res.append(vocab)
    return res
    
def vocab_ids(path,start_ids):
    '''
    读取字典文件（utf-8格式），一行为一个词。
    返回有序字典。
    
    start_ids表示索引开始的数字，一般选0或1。
    '''
    assert os.path.exists(path)
    with open(path,"r",encoding="utf") as f:
        lines=f.readlines()
    vocab=[]
    for line in lines:
        if line[-1]=="\n":
            vocab.append((line[:-1],start_ids))
        elif line!="":
            vocab.append((line,start_ids))
        start_ids+=1
    vocab=OrderedDict(vocab)
    return vocab

def tokenize(text,ischar,stops=[]):
    '''
    对文本进行分词。
    返回分过词的列表。
    
    text需要分词的文本。
    ischar=True表示按字分词，否则按词。
    stops如果为空，则不需要删除停用词，否则删去停用词。
    '''
    
    def _cut(text,ischar):
        if ischar:
            return list(text)
        else:
            return jieba.lcut(text)
        
    vocabs=_cut(text,ischar)
    vocabs=del_stops(vocabs,stops)
    return vocabs

def tokens_to_ids(tokens,vocab,skip,empty=0):
    '''
    tokens是一个含有分过词的list。
    vocab是词汇表的索引字典（有序字典）。
    skip=True表示是否跳过没有索引的词汇，不加入结果中。
    empty：如果skip是false，则不跳过索引词汇，不存在词汇的索引用empty填充。
    '''
    res=[]
    for token in tokens:
        if token in vocab.keys():
            res.append(vocab[token])
        elif not skip:
            res.append(empty)
    return res

# def main():
#     from general import vocab
    
#     dataset_dir=r"./data/dataset"
#     vocab_path=r"./data/vocab.txt"
#     stop_path=r"./data/stop.txt"
    
#     # 建立词典
#     vocab_l=vocab.files_vocab(dataset_dir)
#     vocab_l=["[CLS]","[SEP]"]+vocab_l
#     vocab.write_vocab(vocab_l,vocab_path)
    
#     # 删除停用词
#     stops=vocab.read_vocab(stop_path)
#     print(vocab.del_stops(["不过",",","秦始皇","a"],stops))
    
if __name__=="__main__":
    pass