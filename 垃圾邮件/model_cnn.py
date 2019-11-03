import os
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.callbacks import Callback
import matplotlib.pyplot as plt
def savepickle(name,data):
    output=open('%s.pkl'%(name),'wb')
    pickle.dump(data,output)
    output.close()
def loadpickle(name):
    pkl_file=open('%s.pkl'%(name),'rb')
    datal=pickle.load(pkl_file)
    return datal
#数据可视化时显示相关结果
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict,average='micro')
        _val_recall = recall_score(val_targ, val_predict,average='micro')
        _val_precision = precision_score(val_targ, val_predict,average='micro')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('- val_f1: %.4f - val_precision: %.4f - val_recall: %.4f'%(_val_f1, _val_precision, _val_recall))
        return
#读取数据
df_train = loadpickle('train_data') 
df_test = loadpickle('test_data')
#print(df_train.describe())
#print(df_test.describe())
#显示正例和负例个数
ham_num=len([x for x in df_train['label'] if x==0])
spam_num=len([x for x in df_train['label'] if x==1])
print('ham:',ham_num,'spam:',spam_num)

#进行数字表示
tokenizer = Tokenizer(nb_words=20000,filters='!"#$%&()*+,-./:;<=>?[\]^_`{|}~\t\n')#返回的是频率最高的20000个词
#将几个属性连接成一个句子
train_texts = df_train['from'] +' '+ df_train['to'] +' '+ df_train['subject']
test_texts = df_test['from'] +' '+ df_test['to'] +' '+ df_test['subject']
#将2个数据列表转换为一个数据列表
texts=train_texts.append(test_texts)
tokenizer.fit_on_texts(texts)#训练的文本列表
train_data = tokenizer.texts_to_sequences(train_texts)  # 传入样本列表，形成[23,3,2,56]，数值为词的编号
test_data = tokenizer.texts_to_sequences(test_texts)
word_index = tokenizer.word_index
#统计句子词的长度
'''len_data=[]
for i in train_data:
    len_data.append(len(i))
for i in test_data:
    len_data.append(len(i))
len_data=pd.Series(len_data)
len_data.plot()'''
#根据图像可以知道平均长度大致为30
MAX_SENTENCE_LENGTH = 25
#padding                                             #补零从结尾补0，截断从结尾截断
train_data = pad_sequences(train_data, maxlen=MAX_SENTENCE_LENGTH, padding='post', truncating='post')
test_data = pad_sequences(test_data, maxlen=MAX_SENTENCE_LENGTH, padding='post', truncating='post')
#划分训练集测试集
indices = np.arange(train_data.shape[0])  # 生成一个一维向量[0,...,206247]的数组
np.random.shuffle(indices)
train_data=train_data[indices]
label = to_categorical(np.asarray(df_train['label']))[indices]
X_train = train_data[:40000]
Y_train =label[:40000]
X_test = train_data[40000:]
Y_test =label[40000:]
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.models import Sequential
#建立模型
metrics = Metrics()
#单词的维度
output_dim =64
lstm_out = 196
filters = 250
kernel_size = 3
hidden_dims = 250
model = Sequential()

# Embedding layer把输入的每个词汇数据转换成向量，由这些向量组成一个矩阵。
# input_dim为词汇表大小，output_dim为输出的维度（超参数，可调），
# input_length为输入的句子长度

model.add(Embedding(input_dim=20000, output_dim=output_dim, input_length =X_train.shape[1]))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#模型编译
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
model.summary()

#开始训练
batch_size = 256
history=model.fit(X_train, Y_train, epochs = 3, batch_size=batch_size,validation_data=(X_test,Y_test),callbacks=[metrics])
# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

#模型评估
'''score,acc = model.evaluate(X_test, Y_test, verbose = 1, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))'''
ans = model.predict(test_data,batch_size)
#将答案写入
filename='yuhonghai201763009.txt'
with open(filename,'w') as f:
    f.write('TYPE ID\n')
    for i in range(9999):
        if ans[i][1]>0.5:
            f.write('spam ')
        else:
            f.write('ham ')
        text=df_test['docno'][i]
        num1=text[7:10]
        num2=text[11:14]
        f.write('../Data/'+num1+'/'+num2+'\n')

            
            



