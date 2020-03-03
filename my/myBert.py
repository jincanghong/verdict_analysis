#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import myVocab

def single_input(txt,max_seq_len,vocab_dict,ischar,hascls,hassep,stops=[]):
    '''
    输入txt，按最大长度将文本进行裁切。获得格式化的单个文本。
    
    params:
        txt:文本;
        max_seq_len:文本最大长度;
        vocab_dict：词典；
        ischar=True表示分词成一个个字，否则正常分词；
        cls：表示开头是否包含[CLS]
        sep: 表示结尾是否包含[SEP]
    返回三个列表：tokens_ids,tokens_mask,tokens_seg
    
    bert的输入一般如下所示：
        (a) 两个句子:
        tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        (b) 单个句子:
        tokens:   [CLS] the dog is hairy . [SEP]
        type_ids: 0     0   0   0  0     0 0
        
        type_ids 在源码中对应的是 segment_ids，主要用于区分第一个第二个句子。
        第一个句子为0，第二个句子是1。在预训练的时候会添加到单词的的向量中，但这个不是必须的
        因为[SEP] 已经区分了第一个句子和第二个句子。但type_ids 会让学习变的简单
        
        mask的作用是处理含有文字的内容，不处理填充[PAD]的内容。
    '''
    assert "[PAD]" in vocab_dict.keys()
    if hascls:
        assert "[CLS]" in vocab_dict.keys()
    if hassep:
        assert "[SEP]" in vocab_dict.keys()
    
    split_tokens=myVocab.tokenize(txt,ischar=ischar,stops=stops)
    tokens_ids=myVocab.tokens_to_ids(split_tokens,vocab_dict,skip=True)
    # 添加cls和sep
    if hascls:
        tokens_ids=[vocab_dict["[CLS]"]]+tokens_ids[:max_seq_len-1]
    else:
        tokens_ids=tokens_ids[:max_seq_len]
    if hassep:
        if len(tokens_ids)==max_seq_len:
            tokens_ids[-1]=vocab_dict["[SEP]"]
        else:
            tokens_ids+=[vocab_dict["[SEP]"]]
    # 统一到相同长度
    tokens_mask=[1]*len(tokens_ids)+[0]*(max_seq_len-len(tokens_ids))
    tokens_seg=[0]*max_seq_len
    tokens_ids+=[vocab_dict["[PAD]"]]*(max_seq_len-len(tokens_ids))
    return tokens_ids,tokens_mask,tokens_seg

def double_input(txt1,txt2,max_seq_len,vocab_dict,ischar,stop=[]):
    '''
    输入txt，将文本拼接起来，中间以[SEP]分割，再按最大长度将文本进行裁切。获得格式化的拼接文本。
    
    params:
        txt1,txt2:两段文本;
        max_seq_len:文本最大长度;
        vocab_dict：词典；
        ischar=True表示分词成一个个字，否则正常分词；
    返回三个列表：tokens_ids,tokens_mask,tokens_seg
    
    bert的输入一般如下所示：
        tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        
        type_ids 在源码中对应的是 segment_ids，主要用于区分第一个第二个句子。
        第一个句子为0，第二个句子是1。在预训练的时候会添加到单词的的向量中，但这个不是必须的
        因为[SEP] 已经区分了第一个句子和第二个句子。但type_ids 会让学习变的简单
        
        mask的作用是处理含有文字的内容，不处理填充[PAD]的内容。
    '''
    raise NotImplementedError()

def get_bert_word_vectors():
    raise NotImplementedError()

def get_bert_sent_vectors():
    raise NotImplementedError()

if __name__=="__main__":
    vocab_dict=myVocab.vocab_ids(r"E:\code\judgement\code\bert\chinese_L-12_H-768_A-12\vocab.txt",start_ids=0)
    print(single_input(":文本最大长度;ischar=True表示分词成一个个字表示分词成一个个字表示分词成一个个字",
                       max_seq_len=10,vocab_dict=vocab_dict,
                       ischar=True,hascls=True,hassep=True))