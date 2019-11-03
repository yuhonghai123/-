import os
import pandas as pd
import numpy as np
import pickle
import time
import jieba
def savepickle(name,data):
    output=open('./%s.pkl'%(name),'wb')
    pickle.dump(data,output)
    output.close()
def loadpickle(name):
    pkl_file=open('./%s.pkl'%(name),'rb')
    datal=pickle.load(pkl_file)
    return datal
def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False
files=os.listdir('./train/Data')
#print(files)
file_instance='./train/Data/001/004'
label_file='./index.txt'
DOCNO=[]
FROM=[]
TO=[]
SUBJECT=[]  
LABEL=[]
#添加label
with open(label_file,'r')as f1:
    for line in f1:
        if len(line)!=0:
            if(line[0]=='h'):
                LABEL.append(0)
            elif(line[0]=='s'):
                LABEL.append(1)
'''with open(file_instance,'r') as f:
    for line in f:
        if "<DOCNO>" in line:
            s=line.strip()[7:-8]
            if len(s)!=0:
                DOCNO.append(s)
            else:
                DOCNO.append("nodoc")
        elif "<FROM>" in line:
            s=line.strip()[6:-7]
            if len(s)!=0:
                FROM.append(s)
            else:
                FROM.append("nosourse")
        elif "<TO>" in line:
            s=line.strip()[4:-5]
            if len(s)!=0:
                TO.append(s)
            else:
                TO.append("nodestination")
        elif "<SUBJECT>" in line:
            s=line.strip()[9:-10]
            if len(s)!=0:
                SUBJECT.append(s)
            else:
                SUBJECT.append("nosubject")'''
start =time.clock()

#从文件中提取特征from,to,subject
for bigfile in files:#500个
    smallfile=os.listdir('./train/Data/'+bigfile)
    for file in smallfile:#100个
        filename='./train/Data/'+bigfile+'/'+file
        with open(filename,'r',encoding='ansi') as f:
            for line in f:
                if "<DOCNO>" in line:
                    s=line.strip()[7:-8]
                    if len(s)!=0:
                        DOCNO.append(s)
                    else:
                        DOCNO.append("nodoc")
                elif "<FROM>" in line:
                    s=line.strip()[6:-7]
                    if len(s)!=0:
                        FROM.append(s)
                    else:
                        FROM.append("nosourse")
                elif "<TO>" in line:
                    s=line.strip()[4:-5]
                    if len(s)!=0:
                        TO.append(s)
                    else:
                        TO.append("nodestination")
                elif "<SUBJECT>" in line:
                    s=line.strip()[9:-10]
                    if len(s)!=0:
                        SUBJECT.append(s)
                    else:
                        SUBJECT.append("nosubject")
end=time.clock()
print(end-start)

#由于文件有缺失，所以要找到实际的label，不能直接一一对应
REAL_LABEL=[]
for term in DOCNO:
    a=int(term[7:10])
    b=int(term[11:14])
    REAL_LABEL.append(LABEL[(a-1)*100+b-1])
    
split_from = []
split_to = []
split_subject = []
#split_subject = [(' ').join(jieba.lcut(s)) for s in SUBJECT if is_contains_chinese(s)]
for s in FROM:
    s=s.replace('@',' ')
    s=s.replace('.',' ')
    split_from.append(s)
for s in TO:
    s=s.replace('@',' ')
    s=s.replace('.',' ')
    split_to.append(s)
for s in SUBJECT:
    if is_contains_chinese(s):
        split_subject.append((' ').join(jieba.lcut(s)))
    else:
        split_subject.append(s)
df=pd.DataFrame()
df1=pd.DataFrame()
df['docno'] = DOCNO
df['from'] = split_from
df['to'] = split_to
df['subject'] = split_subject
df['label'] = REAL_LABEL
savepickle('train_data', df)
df1['docno'] = DOCNO
df1['from'] = FROM
df1['to'] = TO
df1['subject'] = SUBJECT
       
            
            
                    
            
            
            
            