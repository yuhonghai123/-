import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score,f1_score
def savepickle(name,data):
    output=open('./%s.pkl'%(name),'wb')
    pickle.dump(data,output)
    output.close()
def loadpickle(name):
    pkl_file=open('./%s.pkl'%(name),'rb')
    datal=pickle.load(pkl_file)
    return datal
#下载数据
X=loadpickle('X')
Y=loadpickle('LABEL')
#构建模型
train_x=X[:40000]
train_y=Y[:40000]
test_x=X[40000:]
test_y=Y[40000:]
limit_X=[x[:20] for x in X]
limit_train_x=limit_X[:40000]
limit_test_x=limit_X[40000:]
#构建一个流，提取词频逆文档频率，并将其输入到贝叶斯分类器中
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(limit_train_x, train_y)
#预测test集
ans=model.predict(limit_test_x)
print('precision',precision_score(test_y,ans))
print('f1_score',f1_score(test_y,ans))

