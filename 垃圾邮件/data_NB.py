import os
import pickle
#定义存取函数，方便每次直接读取数据，而不需要再次处理
def savepickle(name,data):
    output=open('./%s.pkl'%(name),'wb')
    pickle.dump(data,output)
    output.close()
def loadpickle(name):
    pkl_file=open('./%s.pkl'%(name),'rb')
    datal=pickle.load(pkl_file)
    return datal
files=os.listdir('./train/Data')
label_file='./index.txt'
#保存每个邮件的内容
X =[]
#保存每个邮件的标签
LABEL=[]
#读取标签
with open(label_file,'r')as f1:
    for line in f1:
        if len(line)!=0:
            if(line[0]=='h'):
                LABEL.append(0)
            elif(line[0]=='s'):
                LABEL.append(1)
#读取邮件的内容
for bigfile in files:#500个
    smallfile=os.listdir('./train/Data/'+bigfile)
    for file in smallfile:#100个
        filename='./train/Data/'+bigfile+'/'+file
        with open(filename,'r',encoding='ansi') as f:
            X.append(f.read())
#由于train集的邮件有缺失，需要筛选对应的标签，存储在real_label中
REAL_LABEL=[]
#找到相应的标签
for term in X:
    a=int(term[20:23])
    b=int(term[24:27])
    REAL_LABEL.append(LABEL[(a-1)*100+b-1])
#保存数据
savepickle('X',X)
savepickle('LABEL',REAL_LABEL)

            
            