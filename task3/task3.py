#coding=utf-8
import csv
import numpy as np
import pandas as pd
from pandas import Series
import sys
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

vali_author_list=[]


def data_loader(train_path='train.csv',paper_path='papers.txt',vali_path='task3_test_final.csv'):
    '''
    加载函数，将数据转化为模型需要的数据
    '''
    print ("building data...\n")
    with open(train_path, 'r') as f:
        incsv = csv.reader(f)
        header = next(incsv)
        author_idx = header.index('AUTHOR')
        citation_idx = header.index('CITATION')
        author_list=[]
        citation_list=[]
        for line in incsv:
            author = line[author_idx].strip()
            citation = line[citation_idx].strip()
            author_list.append(author)
            citation_list.append(citation)
        assert len(author_list)==len(citation_list)
        author_to_citation= dict(zip(author_list,citation_list))

    with open(paper_path, 'r',encoding='utf-8') as f:
        """
        作者合作关系字典(author_to_relation)：{'作者名'：['合作人1'，'合作人2'，...]}
        作者合作关系引用字典(author_to_relation_citation)：{'作者名'：['num1'，'num2'，...]}
        """
        author_to_relation=dict()
        author_to_relation_citation=dict()
        num=0
        for line in f.readlines():
            num=num+1
            print(num)
            name_list=[]
            if '#@' in line:
                name_list=line[2:].strip('\n').split(',')
                author_to_relation=add_to_dict(author_to_relation,name_list)
            else:
                continue
        assert len(author_to_relation.keys())>0
        print ("train data")
        i=0
        for key,value in author_to_relation.items():
            i+=1
            if key in author_to_citation.keys():
                temp=[author_to_citation[x] for x in value if x in author_to_citation.keys()]
                if len(temp)>0:
                    author_to_relation_citation[key]=temp
            sys.stdout.write('generated:{0}/total:{1}\r'.format(i,len(author_to_relation.keys())))
            sys.stdout.flush()
    train_X_data,train_y_data=conver_to_model_data(author_to_relation_citation,author_to_citation)
    assert len(train_X_data)==len(train_y_data)


    with open(vali_path, 'r') as f:
        global vali_author_list
        Vali_author_to_relation_citation=[]
        incsv = csv.reader(f)
        header = next(incsv)
        author_idx = header.index('AUTHOR')
        for line in incsv:
            author = line[author_idx].strip()
            vali_author_list.append(author)
        assert len(vali_author_list)==300000
        i=0
        print ("\nvali data")
        for author in vali_author_list:
            i+=1
            if author in author_to_relation.keys():
                author_relation=author_to_relation[author]
                temp=[author_to_citation[x] for x in author_relation if x in author_to_citation.keys()]
                if len(temp)>0:
                    Vali_author_to_relation_citation.append(temp)
                else:
                    Vali_author_to_relation_citation.append(np.zeros(8))
            else:
                Vali_author_to_relation_citation.append(np.zeros(8))

            sys.stdout.write('generated:{0}/total:{1}\r'.format(i,len(vali_author_list)))
            sys.stdout.flush()
    assert len(Vali_author_to_relation_citation)==len(vali_author_list)
    vali_X_data= conver_to_model_data(Vali_author_to_relation_citation,author_to_citation,is_train=False)
    print ("\nbuild data done...")
    return  train_X_data,train_y_data,vali_X_data

def conver_to_model_data(author_to_relation_citation='',author_to_citation='',is_train=True):
    '''
    转化数据处理函数：将每个作者的合作者引用量list分别取：最大值，最小值，中位数，均值，方差，1/4分位数，3/4分位数，偏度作为模型特征
    其预测结果与本作者真实引用量做拟合
    '''
    if is_train:
        X_data=[[np.max(np.array(value,dtype=float)),np.min(np.array(value,dtype=float)),np.median(np.array(value,dtype=float)),\
        np.mean(np.array(value,dtype=float)),np.var(np.array(value,dtype=float)),np.percentile(np.array(value,dtype=float),25),\
        Series(np.array(value,dtype=float)).skew()] for value in author_to_relation_citation.values()]
        y_data=[author_to_citation[x] for x in author_to_relation_citation.keys()]
        return np.nan_to_num(np.array(X_data,dtype=float)),np.nan_to_num(np.array(y_data,dtype=float))
    else:
        X_data=[[np.max(np.array(value,dtype=float)),np.min(np.array(value,dtype=float)),np.median(np.array(value,dtype=float)),\
        np.mean(np.array(value,dtype=float)),np.var(np.array(value,dtype=float)),np.percentile(np.array(value,dtype=float),25),\
        Series(np.array(value,dtype=float)).skew()] for value in author_to_relation_citation]
        return np.nan_to_num(np.array(X_data,dtype=float))

def add_to_dict(author_to_relation,name_list):
    for author in name_list:
        if author in author_to_relation.keys():
            author_to_relation[author].extend([x for x in name_list if x!=author])
            author_to_relation[author]=list(set(author_to_relation[author]))#去重
        else:
            author_to_relation[author]=[x for x in name_list if x!=author]
    return author_to_relation



if __name__ == '__main__':
    train_X_data,train_y_data,vali_X_data=data_loader()

    X_train, X_test, y_train, y_test = train_test_split(train_X_data, train_y_data, test_size=0.25, random_state=33)
    # GBR=GradientBoostingRegressor()
    # GBR=GBR.fit(X_train, y_train)
    GBR=joblib.load('GBR.model')
    print ("GBR train score is {}".format(GBR.score(X_test, y_test)))
    #
    # #start pred task#
    # GBR=GBR.fit(X_train, y_train)
    pred=GBR.predict(vali_X_data)
    assert len(pred)==len(vali_author_list)
    pred_data=dict(zip(vali_author_list,np.around(pred.tolist())))
    for key,value in pred_data.items():
        f=open('task3_final.txt','a',encoding='utf-8')
        f.write(key+'    '+str(int(value))+'\n')
    # df=pd.DataFrame(pred_data,columns=['authorname','citation'])
    # df.to_csv("result.csv", sep=' ',index=False)
    print ("compute completed..")
