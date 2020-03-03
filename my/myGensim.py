#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import myVocab,myUtils,myRe
from gensim.models import word2vec  

def gensim_word2vec(sentences,model_path):
    
    '''
    gensim.models.word2vec.Word2Vec(
        sentences=None,size=100,alpha=0.025,window=5,min_count=5,max_vocab_size=None,
        sample=0.001,seed=1, workers=3,min_alpha=0.0001, sg=0,hs=0,negative=5,
        cbow_mean=1,hashfxn=,iter=5,null_word=0,trim_rule=None,sorted_vocab=1,batch_words=10000)
    
    sentences: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出。对于大语料集，建议使用BrownCorpus,Text8Corpus或lineSentence构建。
    size: 词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，视语料库的大小而定。
    alpha： 是初始的学习速率，在训练过程中会线性地递减到min_alpha。
    window：即词向量上下文最大距离，skip-gram和cbow算法是基于滑动窗口来做预测。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。对于一般的语料这个值推荐在[5,10]之间。
    min_count:：可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5。
    max_vocab_size: 设置词向量构建期间的RAM限制，设置成None则没有限制。
    sample: 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)。
    seed：用于随机数发生器。与初始化词向量有关。
    workers：用于控制训练的并行数。workers控制训练的并行，此参数只有在安装了Cpython后才有效，否则只能使用单核。
    min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每 轮的迭代步长可以由iter，alpha， min_alpha一起得出。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。
    sg: 即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。
    hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。
    negative:如果大于零，则会采用negativesampling，用于设置多少个noise words（一般是5-20）。
    cbow_mean: 仅用于CBOW在做投影的时候，为0，则采用上下文的词向量之和，为1则为上下文的词向量的平均值。默认值也是1,不推荐修改默认值。
    hashfxn： hash函数来初始化权重，默认使用python的hash函数。
    iter: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。
    trim_rule： 用于设置词汇表的整理规则，指定那些单词要留下，哪些要被删除。可以设置为None（min_count会被使用）。
    sorted_vocab： 如果为1（默认），则在分配word index 的时候会先对单词基于频率降序排序。
    batch_words：每一批的传递给线程的单词的数量，默认为10000。
    '''
    print("开始训练word2vec")
    model=word2vec.Word2Vec(sentences,sg=1,size=200,window=6,negative=5,sample=0.001,hs=1,workers=4,min_count=3)
    print("结束训练word2vec，保存模型")
    model.save(model_path)
    print("保存模型完毕")
    return
    
def load_word2vec(model_file):
    model=word2vec.Word2Vec.load(model_file)
    #print(model.wv.vocab)
    #print(model.similar_by_word("吸毒",topn=10))
    return model
    
def format_file(file,stops):
    '''
    读取文档，格式化文档，并返回纯文本。
    纯文本格式为：一行一句话，一句话里的词都用空格分割开了。
    '''
    with open(file,"r",encoding="utf-8") as f:
        lines=f.readlines()
    res=[]
    for line in lines:
        line=myVocab.tokenize(line.strip(),ischar=False,stops=stops)
        line=" ".join(line)
        res.append(line)
    res="\n".join(res)
    res=myRe.del_repeat(res,"\n")
    return res

def format_dir(dir,dest_file,stops):
    '''
    将文件夹里的所有文件都改变格式，合并成一个文本，存在dest_file文档里。
    '''
    for file in os.listdir(dir):
        print(file)
        txt=format_file(os.path.join(dir,file),stops)
        with open(dest_file,"a",encoding="utf-8") as f:
            f.write(txt+"\n")
    return
    
if __name__ == "__main__":
    #stops=myUtils.get_list_by_line('../data/stop.txt')
    #format_dir(r"E:\code\judgement\ref\fulldatatxt","../data/cbow.txt",stops)
    
    # with open("../data/cbow.txt","r",encoding="utf-8") as f:
    #     lines=f.readlines()
    # lines=[line.strip().split(" ") for line in lines]
    # print(lines[:10])
    
    model_path="model.work2vec"
    # gensim_word2vec(lines,model_path)
    model=word2vec.Word2Vec.load(model_path)
    #print(model.wv.vocab)
    print(model.similar_by_word("重大损失",topn=10))
    pass