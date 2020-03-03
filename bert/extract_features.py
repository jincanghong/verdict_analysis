import sys
sys.path.append("../")
sys.path.append("../my/")
sys.path.append("./bert/")

import os
import re
import jieba
import random
import numpy as np
import tensorflow as tf
import pandas as pd

from bert import modeling
from my import myVocab,myUtils,myRe,myMath,myLoss
from bert.tokenization import load_vocab,BasicTokenizer,WordpieceTokenizer,convert_tokens_to_ids

class CharTokenizer(object):
    """Runs end-to-end tokenziation."""
    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
    
    def tokenize(self, text):
        split_tokens = []
        #print("\nself.basic_tokenizer.tokenize(text):\n",self.basic_tokenizer.tokenize(text))
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in token:
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_tokens_to_ids(self.vocab,tokens)


def get_all_vec(nofuzzy_src,fuzzy_src,dest_dir,max_seq_len=512):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    
    input_ids=tf.placeholder(tf.int32,shape=[None,None],name='input_ids')
    input_mask=tf.placeholder(tf.int32,shape=[None,None],name='input_masks')
    segment_ids=tf.placeholder(tf.int32,shape=[None,None],name='segment_ids')
    
    # 初始化BERT
    bert_config=modeling.BertConfig.from_json_file("./bert/chinese_L-12_H-768_A-12/bert_config.json")
    init_check_point="./bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
    vocab_file="./bert/chinese_L-12_H-768_A-12/vocab.txt"
    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False)

    # 加载bert模型
    tvars = tf.trainable_variables()
    (assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_check_point)
    tf.train.init_from_checkpoint(init_check_point, assignment)
    
    vectors=model.all_encoder_layers[-2]
    aver_feature=tf.reduce_mean(vectors,axis=[1])
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 分词
        tokenizer=CharTokenizer(vocab_file=vocab_file)
        
        # nofuzzy
        print("Bert:process nofuzzy")
        input_res=pd.DataFrame(columns=["姓名","判刑月份","毒品数量",]+[str(i) for i in range(max_seq_len)])
        aver_res=pd.DataFrame(columns=["姓名","判刑月份","毒品数量",]+[str(i) for i in range(768)])
        # read file
        with open(nofuzzy_src,"r",encoding="utf-8") as f:
            lines=f.readlines()
        for line in lines:
            line=line.strip().split("\t")
            if len(line)==5:#id,name,sent,gram,txt
                split_tokens=tokenizer.tokenize(line[-1])
                split_tokens=["[CLS]"]+split_tokens[:max_seq_len-2]+["[SEP]"]
                word_ids=tokenizer.convert_tokens_to_ids(split_tokens)
                word_mask=[1]*len(word_ids)
                word_segment_ids=[0]*len(word_ids)
                fd={input_ids:[word_ids],input_mask:[word_mask],segment_ids:[word_segment_ids]}
                aver_output=sess.run(aver_feature,feed_dict=fd)
                # save into pd
                input_res.loc[line[0]]=[line[1],line[2],line[3]]+word_ids+[0]*(max_seq_len-len(word_ids))
                aver_res.loc[line[0]]=[line[1],line[2],line[3]]+aver_output[0].tolist()
                # if line[0]=="data-3":
                #     print(input_res)
                #     print(aver_res)
                #     input("nofuzzy...")
                #     break
        input_res.to_csv(os.path.join(dest_dir,"./nofuzzy.input.csv"),index=True,header=True)
        aver_res.to_csv(os.path.join(dest_dir,"./nofuzzy.averpool.csv"),index=True,header=True)
        
        # fuzzy
        print("Bert:process fuzzy")
        input_res=pd.DataFrame(columns=["姓名","判刑月份","毒品数量",]+[str(i) for i in range(max_seq_len)])
        aver_res=pd.DataFrame(columns=["姓名","判刑月份","毒品数量",]+[str(i) for i in range(768)])
        # read file
        with open(fuzzy_src,"r",encoding="utf-8") as f:
            lines=f.readlines()
        for line in lines:
            line=line.strip().split("\t")
            if len(line)==5:#id,name,sent,gram,txt
                split_tokens=tokenizer.tokenize(line[-1])
                split_tokens=["[CLS]"]+split_tokens[:max_seq_len-2]+["[SEP]"]
                word_ids=tokenizer.convert_tokens_to_ids(split_tokens)
                word_mask=[1]*len(word_ids)
                word_segment_ids=[0]*len(word_ids)
                fd={input_ids:[word_ids],input_mask:[word_mask],segment_ids:[word_segment_ids]}
                aver_output=sess.run(aver_feature,feed_dict=fd)
                # save into pd
                input_res.loc[line[0]]=[line[1],line[2],line[3]]+word_ids+[0]*(max_seq_len-len(word_ids))
                aver_res.loc[line[0]]=[line[1],line[2],line[3]]+aver_output[0].tolist()
                # if line[0]=="data-3":
                #     print(input_res)
                #     print(aver_res)
                #     input("fuzzy...")
                #     break
        input_res.to_csv(os.path.join(dest_dir,"./fuzzy.input.csv"),index=True,header=True)
        aver_res.to_csv(os.path.join(dest_dir,"./fuzzy.averpool.csv"),index=True,header=True)
        
        print("Bert:finished")
    return

if __name__ == "__main__":
    get_all_vec("../dataset/data.ori.txt","../dataset/data.fuzzy.txt","./data")
