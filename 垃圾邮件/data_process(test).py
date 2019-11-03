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
files=os.listdir('./test/Data')
DOCNO=[]
FROM=[]
TO=[]
SUBJECT=[]
for bigfile in files:#500个
    smallfile=os.listdir('./test/Data/'+bigfile)
    for file in smallfile:#100个
        filename='./test/Data/'+bigfile+'/'+file
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
split_from = []
split_to = []
split_subject = []
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
df['docno'] = DOCNO
df['from'] = split_from
df['to'] = split_to
df['subject'] = split_subject
savepickle('test_data', df)