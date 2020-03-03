
import sys
sys.path.append("./my")

import re
import os
import jieba
import myUtils,myVocab,myRe

def fuzzy(src_file,dest_file,stops_path,freqs_path):
    def _fuzzy(txt):
        # 替换顺序不能变
        txt=re.sub("(第?[零〇一二两三四五六七八九十0123456789]{1,3}个?[月日])|([零〇一二两三四五六七八九十0123456789]{4,5}年)|(第?[零〇一二两三四五六七八九十百千0123456789]{1,5}[条款])","",txt)
        txt=re.sub("[零〇一二两三四五六七八九十0123456789]{2,3}年","较长年份",txt)
        txt=re.sub("[零〇一二两012]年","较短年份",txt)
        txt=re.sub("[三四五六七八九十3456789(10)]年","中等年份",txt)
        #print("txt",txt)
        txt=re.sub("第?[1一]次","较少次",txt)
        txt=re.sub("第?[0-9零〇一二两三四五六七八九十]{1,}次","较多次",txt)
        #print(txt)
        txt=re.sub("([0-9]{5,}元)|([一二两三四五六七八九十]万[零〇一二两三四五六七八九十千百]*元)|([5-9][0-9]{3,3}元)|([五六七八九]千[零〇一二两三四五六七八九十百]+元)","较大金额",txt)
        txt=re.sub("([1-4][0-9]{3,3}元)|([一二两三四五]千[零〇一二两三四五六七八九十百]*元)","中等金额",txt)
        txt=re.sub("[0-9]{1,3}元|[零〇一二两三四五六七八九十百]+元","较少金额",txt)
        
        # 
        regex=["[0-9]+(\\.[0-9]+)?(克|元)","[零〇一二两三四五六七八九十百千]+[元克]","(出生于|户籍地)(.+?)[,，.。]","车牌号(.*)[0-9A-Za-z]",
        "[0-9零〇一二两三四五六七八九十]{1,4}年","[0-9零〇一二两三四五六七八九十][0-9零〇一二两三四五六七八九十]?个?[月日]","在(.+?)(宾馆|酒店|会所|家中|大道|街|饭店)","身份证[号]?[A-Za-z0-9]{18}",
        "(OPPO|苹果|三星)[牌]?手机"]
        for reg in regex:
            txt=re.sub(reg,"",txt)
        
        return txt
    
    res=[]
    stops=myUtils.get_list_by_line(stops_path)
    freqs=myUtils.json_file_to_dict(freqs_path)
    with open(src_file,"r",encoding="utf-8") as f:
        lines=f.readlines()
    for line in lines:
        if line.strip()!="":
            line=line.strip().split("\t")
            # 处理文本
            txt=_fuzzy(line[-1])
            txt_list=myVocab.tokenize(txt,ischar=False,stops=stops)
            txt_list=myVocab.filter_freq(txt_list,freqs,4,1e12,skip=False)
            txt_list=myRe.filter_location(txt_list)
            txt="".join(txt_list)
            txt=re.sub("，+","，",txt)
            try:
                if txt[0]=="，":
                    txt=txt[1:]
                if txt[-1]=="，":
                    txt=txt[:-1]
            except:
                print(str(lines),lines[3],txt_list)
                input(line)
            line[-1]=txt
            res.append("\t".join(line))
    # 写入文件
    myUtils.write_text_into_file("\n".join(res),dest_file)
    return


if __name__ == "__main__":
    print("pre-precessing")
    fuzzy("./dataset/data.ori.txt","./dataset/data.fuzzy.txt","./dataset/judge_stop.txt","./dataset/judgement_freq.json")