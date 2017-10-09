# coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer
import  numpy as np
import heapq
import pickle
import time
author_papers=dict()
# {'name': papers_str}

def add_paper_str(name_list,paper_str):
    global author_papers
    for name in name_list:
        if name in author_papers:
            author_papers[name]+=' '+paper_str
        else:
            author_papers[name]=paper_str

def add(list_2):
    name_list = []
    paper_str = ''
    for line in list_2:
        if '#@' in line:
            name_list = line[2:].strip('\n').split(',')
            continue
            # print line
        if '#*' in line:
            paper_str = line[2:-1].replace(' in ',' ').replace(' the ',' ').replace(' of ',' ').\
            replace(' on ',' ').replace(' for ',' ').replace(' with ',' ')\
            .replace(' and ',' ').replace(' and ',' ')
            continue
        if len(name_list) != 0 and paper_str != '':
            add_paper_str(name_list, paper_str)





def load():
    num=0
    with open('papers.txt','r',encoding='utf-8') as fp:
        list_2=[]
        for line in fp.readlines():
            num=num+1
            print(num)
            list_2.append(line)
            if line == '\n':
                add(list_2)
                list_2 = []

    print ('load done')

def cos_sim(v1,v2):
    vec1=np.array(v1.toarray())
    vec2=np.array(v2.toarray())
    return np.linalg.norm(vec1 - vec2)


def find_max_idx(sim_list,max_value):
    idx=sim_list.index(max_value)
    sim_list[idx]=10000
    return idx



def find_to_authors(papers_vec,new_vec):
    sim_list=list(map(lambda x:cos_sim(x,new_vec[0]),papers_vec))
    k_maxs=list(map(lambda x:find_max_idx(sim_list,x), heapq.nsmallest(6000, sim_list)))
    # print(heapq.nlargest(500, sim_list))


    # k_maxs=heapq.nlargest(5, range(len(sim_list)), np.array(sim_list),sim_list.take)
    return k_maxs


def gettrain():
    train_list=[]
    train_dict={}
    f=open('training.txt','r',encoding='utf-8')
    for line in f.readlines():
        train_list.append(line)
        if line=='\n':
            author=train_list[0].strip('\n')
            interest=train_list[1].strip('\n').split(',')
            train_dict[author]=interest
            train_list=[]
    return  train_dict


def pre_interest(author,vec,papers_vec,idx_to_name,train_dict):
    # fp=open('papers_vec.pkl','rb')
    # author_papers=pickle.load(fp)
    # idx_to_name=dict(zip(range(len(author_papers)),author_papers.keys()))
    # vec = CountVectorizer()
    # papers_vec = vec.fit_transform(author_papers.values())
    # print(type(papers_vec))
    # fp=open('papers_vec.pkl','rb')
    # papers_vec=pickle.load(fp)
    new_author_paper=[author_papers[author]]
    new_vec=vec.transform(new_author_paper)
    k_max_sim=find_to_authors(papers_vec,new_vec)#作者索引
    k_max_sim_names=list(map(lambda x:idx_to_name[x],k_max_sim))#最高相似作者
    # train_dict=gettrain()
    interest_list=[]
    for name in k_max_sim_names:
        if name in train_dict:
            interest_list.extend(train_dict[name])
            if len(interest_list)>5:
                interest_list=interest_list[:5]
                break
    return interest_list


def author_list():
    authors=[]
    f=open('task2_test_final.txt','r',encoding='utf-8')
    for line in f.readlines():
        authors.append(line.strip('\n'))
    return authors

def main():
    load()
    # fp=open('author_papers.pkl','rb')
    # author_papers=pickle.load(fp)
    # print(type(author_papers))
    train_dict=gettrain()
    idx_to_name=dict(zip(range(len(train_dict)),train_dict.keys()))
    vec = CountVectorizer()
    papers_vec = vec.fit_transform([author_papers[x] for x in list(train_dict.keys())])
    num=0
    authors=author_list()
    for author in authors:
        try:
            num=num+1
            print(author,str(num))
            interest_list=pre_interest(author,vec,papers_vec,idx_to_name,train_dict)
            f=open('task2_final.txt','a',encoding='utf-8')
            f.write(author+'    '+'  '.join(interest_list)+'\n')
        except:
            f=open('task2_final.txt','a',encoding='utf-8')
            f.write(author+'    '+'  '.join(['interests1','interests2','interests3','interests4','interests5'])+'\n')


if __name__=='__main__':
    main()
    print('Finshed')
# print(len(author_list()))