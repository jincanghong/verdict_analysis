#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from docx import Document

def doc2docx(src_path,dest_path,log_path):
    '''
    src_path,dest_path同为文件夹或者同为文件。
    
    导入win32com模块，使用Dispatch("Word.Application")打开word应用程序，
    然后再逐个打开doc文件doc = word.Documents.Open(file)，
    使用SaveAs另存文件为"docx"后缀的文件，另存之后，关闭打开的doc文件，再处理下一个，直到结束。
    我们保存新文件的时候，还是保持到原文件路径，只是将后缀改为".docx"，
    "{}x".format(file)指将在原文件名的末尾加上“x”。
    '''
    from win32com import client as wc #导入模块
    
    def _doc2docx(src_file,dest_file,word_obj,log_path):
        try:
            doc = word_obj.Documents.Open(src_file) #打开word文件
            doc.SaveAs(dest_file, 12)#另存为后缀为".docx"的文件，其中参数12指docx文件
            doc.Close() #关闭原来word文件
        except Exception as e:
            print(type(e),e)
            with open(log_path,"a",encoding="utf-8") as f:
                f.write("%s %s\n"%(str(type(e)),str(e)))
        return
    
    if os.path.isfile(src_path) and not os.path.isdir(dest_path):
        word_obj=wc.Dispatch("Word.Application")#打开word应用程序
        print("%s => %s"%(src_path,dest_path))
        _doc2docx(src_path,dest_path,word_obj,log_path)
        word_obj.Quit()
    elif os.path.isdir(src_path) and os.path.isdir(dest_path):
        word_obj=wc.Dispatch("Word.Application")#打开word应用程序
        for file in os.listdir(src_path):
            if file[-4:]==".doc":
                tmpsrc=os.path.join(src_path,file)
                tmpdest=os.path.join(dest_path,file+"x")
                print("%s => %s"%(tmpsrc,tmpdest))
                _doc2docx(tmpsrc,tmpdest,word_obj,log_path)
        word_obj.Quit()
    else:
        raise Exception("\n\tsrc_path is: %s, dest_path is: %s. \n\t源路径和目的路径应该同为文件夹，或者同为文件！"%(src_path,dest_path))
    print("all finished!")
    return

def docx_paras(path):
    res=[]
    doc=Document(path)
    for paragraph in doc.paragraphs:
        res.append(paragraph.text)
    return "\n".join(res)

if __name__ == "__main__":
    # import os
    # os.mkdir("./2017txt")
    # os.mkdir("./2018txt")
    # cnt=1
    # for folder in os.listdir("./2017"):
    #     for file in os.listdir("./2017/"+folder):
    #         print(file)
    #         txt=doc_paras("./2017/"+folder+"/"+file)
    #         with open("./2017txt/"+str(cnt)+".txt","w",encoding="utf-8") as f:
    #             f.write(txt)
    #         cnt+=1
    # for folder in os.listdir("./2018"):
    #     for file in os.listdir("./2018/"+folder):
    #         print(file)
    #         txt=doc_paras("./2018/"+folder+"/"+file)
    #         with open("./2018txt/"+str(cnt)+".txt","w",encoding="utf-8") as f:
    #             f.write(txt)
    #         cnt+=1
    
    # src_folder=r"E:\code\judgement\ref\判决书\2017年走私、贩卖、运输、制造毒品罪"
    # dest_folder=r"E:\code\judgement\ref\fulldata\2017"
    # for folder in os.listdir(src_folder):
    #     src=os.path.join(src_folder,folder)
    #     doc2docx(src,dest_folder,r"C:\Users\shaw\Desktop\doc2docx.log")
    # src_folder=r"E:\code\judgement\ref\判决书\2018年1-6月份毒品刑事案件一审"
    # dest_folder=r"E:\code\judgement\ref\fulldata\2018"
    # for folder in os.listdir(src_folder):
    #     src=os.path.join(src_folder,folder)
    #     doc2docx(src,dest_folder,r"C:\Users\shaw\Desktop\doc2docx.log")
    
    src_folder=r"E:\code\judgement\ref\fulldata\2017"
    dest_folder=r"E:\code\judgement\ref\fulldatatxt"
    cnt=1
    for file in os.listdir(src_folder):
        oldfile=os.path.join(src_folder,file)
        txt=docx_paras(oldfile)
        
        newfile=os.path.join(dest_folder,str(cnt)+".txt")
        with open(newfile,"w",encoding="utf-8") as f:
            f.write(txt)
            
        print("%s => %s"%(oldfile,newfile))
        cnt+=1
        
    src_folder=r"E:\code\judgement\ref\fulldata\2018"
    for file in os.listdir(src_folder):
        oldfile=os.path.join(src_folder,file)
        txt=docx_paras(oldfile)
        
        newfile=os.path.join(dest_folder,str(cnt)+".txt")
        with open(newfile,"w",encoding="utf-8") as f:
            f.write(txt)
            
        print("%s => %s"%(oldfile,newfile))
        cnt+=1
    pass