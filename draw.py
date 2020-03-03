import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd

def draw_bar():
    plt.ylim((0,1500))
    plt.ylabel("Number of judgement reports")
    fig=plt.bar(["Group-1","Group-2","Group-3","Group-4"],[434+186,957+412,528+222,513+216],width=0.3,fc="black")
    for rec in fig:
        plt.text(rec.get_x(),rec.get_height()+5,rec.get_height())
    # print(fig.get_width)
    plt.savefig("./figures/group.png")
    plt.show()

def draw_sentence():
    ###第1组：min(2) max(18) mean(6.679032)
    ###第2组：min(1) max(34) mean(12.494522)
    ###第3组：min(36) max(82) mean(47.054667)
    ###第4组：min(84) max(175) mean(104.362140)
    group=np.array([[2,18,6.679032],[1,34,12.494522],[36,82,47.054667],[84,175,104.362140]])
    x=[1,2,3,4]
    xname=["Group-1","Group-2","Group-3","Group-4"]
    plt.ylabel("Prison Term (months)")
    
    colors=sns.color_palette("Set1",9)
    
    plt.plot(x,group[:,0],marker="s",color=colors[0],label="Minimum")
    plt.plot(x,group[:,1],marker="*",color=colors[1],label="Maximum")
    plt.plot(x,group[:,2],marker="o",color=colors[2],label="Average")
    
    for i in range(3):
        for j in range(4):
            plt.text(x[j]+0.1,group[j][i]-4,"%.2f"%(group[j][i]))
    plt.xticks(x, xname)#, rotation=45)
    
    plt.legend(loc="upper left")
    plt.savefig("./figures/sentence.png")
    plt.show()

def draw_tsne():
    def read_data(data,group):
        if group == 1:
            data = data[data['毒品数量'] == 0]
            data = data[data['判刑月份'] < 36]
        elif group == 2:
            data = data[data['毒品数量'] > 0]
            data = data[data['毒品数量'] < 7]
            data = data[data['判刑月份'] < 36]
        elif group == 3:
            data = data[data['毒品数量'] >= 7]
            data = data[data['毒品数量'] < 10]
            data = data[data['判刑月份'] >= 36]
            data = data[data['判刑月份'] < 84]
        elif group == 4:
            data = data[data['毒品数量'] >= 10]
            data = data[data['毒品数量'] < 50]
            data = data[data['判刑月份'] >= 84]
        else:
            raise Exception()
        return data
    
    def _get_numpy(id_list,data):
        numpy_array=None
        for id in id_list:
            vec=np.array([data.loc[id][3:]],"float32")
            if numpy_array is None:
                numpy_array=vec
            else:
                numpy_array=np.concatenate([numpy_array,vec],axis=0)
        return numpy_array
    
    ids_file="./bert/data/fuzzy.input.csv"
    pool_file="./bert/data/fuzzy.averpool.csv"
    input_save_path="./figures/fuzzy0_input.png"
    output_save_path="./figures/fuzzy0_averpool.png"
    
    # 按lab绘制图像
    perplexity=35
    learning_rate=80.0
    n_iter=3000
    tsne=TSNE(n_components=2,perplexity=perplexity,learning_rate=learning_rate,n_iter=n_iter)
    xmax,xmin=20,-20
    ymax,ymin=20,-20
    #xmax,xmin=None,None
    #ymax,ymin=None,None
    
    # input
    colors=sns.color_palette('Set1',36)
    # sns.palplot(colors)
    # plt.ion()
    # plt.pause(1)
    # plt.close()
    
    # ids
    data = pd.read_csv(ids_file,index_col=0)
    data = read_data(data,group=1)
    # 先获得lab到id的映射，相同的lab的id存储在同一个列表里，即：lab:[id1,id2,...]
    lab2id={}
    for row in data.itertuples():
        sent=row[2]
        if sent in lab2id:
            lab2id[sent].append(row[0])
        else:
            lab2id[sent]=[row[0]]
    labs=list(lab2id.keys())
    labs.sort()
    for lab in labs:
        ids=lab2id[lab]
        lab_numpy=_get_numpy(ids,data)
        if len(lab_numpy)==1:
            continue
        res=tsne.fit_transform(lab_numpy)
        plt.scatter(res[:,0],res[:,1],s=9.0,c=[colors[lab]],label=lab)
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    plt.legend(loc="upper right",facecolor="white")
    plt.savefig(input_save_path)
    plt.ion()
    plt.pause(1)
    plt.close()
    
    # pool
    data = pd.read_csv(pool_file,index_col=0)
    data = read_data(data,group=1)
    # 先获得lab到id的映射，相同的lab的id存储在同一个列表里，即：lab:[id1,id2,...]
    lab2id={}
    for row in data.itertuples():
        sent=row[2]
        if sent in lab2id:
            lab2id[sent].append(row[0])
        else:
            lab2id[sent]=[row[0]]
    labs=list(lab2id.keys())
    labs.sort()
    for lab in labs:
        ids=lab2id[lab]
        lab_numpy=_get_numpy(ids,data)
        if len(lab_numpy)==1:
            continue
        res=tsne.fit_transform(lab_numpy)
        plt.scatter(res[:,0],res[:,1],s=9.0,c=[colors[lab]],label=lab)
    plt.xlim((xmin,xmax))
    plt.ylim((ymin,ymax))
    plt.legend(loc="upper right",facecolor="white")
    plt.savefig(output_save_path)
    plt.ion()
    plt.pause(1)
    plt.close()
    return

if __name__ == "__main__":
    Path("./figures").mkdir(exist_ok=True)
    draw_bar()
    draw_sentence()
    draw_tsne()
    pass