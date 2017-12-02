#coding=utf8
import collections
import math 
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

url='http://mattmahoney.net/dc/'

def  maybe_download(filename,expected_bytes):#验证要不要下载数据文件
    if not os.path.exists(filename):
        filename,_=urllib.urlretrieve(url+filename,filename)
    statinfo=os.stat(filename)
    if statinfo.st_size==expected_bytes:
        print ('Found and verified',filename)
    else :
        print (statinfo.st_size)
        raise Exception(
            'Failed to verify'+filename+'. Can you get to it with a browser?')
    return filename
    
filename=maybe_download('text8.zip', 31344016)
print filename

def read_data(filename):#将数据转化为单词列表
    with zipfile.ZipFile(filename) as f:
        data= tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words=read_data(filename)

print('Data size',len(words))


vocablary_size=50000
def build_dataset(words):
    count=[['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocablary_size-1))#扩充计数表count->(word,CountNum)
    dictionary=dict()
    for word,_ in count:
        dictionary[word]=len(dictionary)    #重新给word分index
    data=list()
    unk_count=0
    for word in words:
        if word in dictionary:
            index=dictionary[word]
        else:
            index=0
            unk_count+=1
        data.append(index)
    count[0][1]=unk_count
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))#把字典反过来变成索引
    return data,count,dictionary,reverse_dictionary

data,count,dictionary,reverse_dictionary=build_dataset(words)

del words   #删除words释放内存


data_index=0
def generate_batch(batch_size,num_skips,skip_window):
    """ generate batch and labels """
    global data_index
    assert batch_size % num_skips==0
    assert num_skips<=2*skip_window
    batch=np.ndarray(shape=[batch_size],dtype=np.int32)
    labels=np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span=2*skip_window+1
    buffer=collections.deque(maxlen=span)   #init a dequence length is span
    for _ in range(span):
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    for i in range(batch_size//num_skips):
        target=skip_window
        target_to_avoid=[skip_window]
        for j in range(num_skips):
            while target in target_to_avoid:
                target=random.randint(0,span-1)
            target_to_avoid.append(target)
            batch[i*num_skips+j]=buffer[skip_window]
            labels[i*num_skips+j,0]=buffer[target]
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    return batch,labels

batch,labels=generate_batch(batch_size=10,num_skips=2,skip_window=1)
for i in range(10):
    print(batch[i],reverse_dictionary[batch[i]],'->',labels[i,0],
          reverse_dictionary[labels[i,0]])

batch_size=128
embedding_size=128
skip_window=1
num_skips=2
valid_size=16
valid_window=100
valid_examples=np.random.choice(valid_window,valid_size,replace=False)
num_samples=64

graph=tf.Graph()
with graph.as_default():
    train_inputs=tf.placeholder(tf.int32, shape=[batch_size])
    train_labels=tf.placeholder(tf.int32,shape=[batch_size,1])
    valid_dataset=tf.constant(valid_examples,dtype=tf.int32)
 
    with tf.device('/cpu:0'):
        #随机生成所有单词的词向量
        embeddings=tf.Variable(
            tf.random_uniform([vocablary_size, embedding_size], -1.0, 1.0)
            )
        #寻找输入的词向量
        embed=tf.nn.embedding_lookup(embeddings, train_inputs)
        #使用NCE Loss 作为优化目标
        nec_weights=tf.Variable(
            tf.truncated_normal([vocablary_size, embedding_size],stddev=1.0/math.sqrt(embedding_size))
            )
        nec_biases=tf.Variable(tf.zeros([vocablary_size]))
        
    loss=tf.reduce_mean(tf.nn.nce_loss(weights=nec_weights, biases=nec_biases,
                                        inputs=embed, labels=train_labels, 
                                        num_sampled=num_samples, num_classes=vocablary_size))
    
    optimizer=tf.train.GradientDescentOptimizer(1.0).minimize(loss)  
    #计算embeddings 的L2范数
    norm=tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
    normalized_embeddings=embeddings/norm
    valid_embeddings=tf.nn.embedding_lookup(
        normalized_embeddings,valid_dataset
        )
    similarity=tf.matmul(
        valid_embeddings,normalized_embeddings,transpose_b=True
        )
    #初始化所有模型参数
    init=tf.global_variables_initializer()
num_steps=10000

with tf.Session(graph=graph) as session:
    init.run()
    print ('Initialized')
    average_loss=0
    for step in range(num_steps):
        batch_inputs,batch_labels=generate_batch(batch_size, num_skips, skip_window)
        feed_dict={train_inputs:batch_inputs,train_labels:batch_labels}
        _,loss_val=session.run([optimizer,loss],feed_dict=feed_dict)
        average_loss+=loss_val
        if step%2000==0:
            if step>0:
                average_loss/=2000
            print("Average loss as step ",step,':',average_loss)
            average_loss=0
        if step % 10000==0:
            sim=similarity.eval()
            for i in range(valid_size):
                valid_word=reverse_dictionary[valid_examples[i]]
                top_k=8
                nearest=(-sim[i,:]).argsort()[1:top_k+1]
                log_str='Nearest to %s:' % valid_word
            for k in range(top_k):
                colse_word=reverse_dictionary[nearest[k]]
                log_str="%s %s ," %(log_str,colse_word)
            print(log_str)
        final_embeddings=normalized_embeddings.eval()
            
    

def plot_with_labels(low_dim_embs,labels,fileaname='tsne.png'):
    assert low_dim_embs.shape[0]>=len(labels),'More labels than emneddings'
    plt.figure(figsize=(18,18))
    for i,label in enumerate(labels):
        x,y=low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(
                    label,xy=(x,y),
                    xytext=(5,2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
    plt.show()
    plt.savefig(fileaname)

tsne=TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
plot_only=100
low_dim_embs=tsne.fit_transform(final_embeddings[:plot_only,:])
labels=[reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)

































    
        
    
