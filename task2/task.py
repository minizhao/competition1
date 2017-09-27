# coding=utf-8
from sklearn.feature_extraction.text import CountVectorizer
import  numpy as np
import heapq
author_papers=dict()
# {'name': papers_str}

def add_paper_str(name_list,paper_str):
    for name in name_list:
        if author_papers.has_key(name):
            author_papers[name]+=' '+paper_str
        else:
            author_papers[name]=paper_str


def load():
    with open('../task3/papers.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            name_list=[]
            paper_str=''
            if '#@' in line:
                name_list=line[2:].strip('\n').split(',')
            if '#*' in line:
                paper_str=line[2:-1]


            if len(name_list)!=0 and paper_str!='':
                add_paper_str(name_list,paper_str)
    print ('load done')




def cos_sim(v1,v2):
    dot_product=v1*v2
    denom=np.linalg(v1)*np.linalg(v2)
    return dot_product/denom


def find_to_authors(papers_vec,new_vec):
    sim_list=list(map(lambda x:cos_sim(x,new_vec),papers_vec))
    k_maxs=heapq.nlargest(5, range(len(sim_list)), np.array(sim_list),sim_list.take)
    return k_maxs


if __name__=='__main__':
    load()
    idx_to_name=zip(range(len(author_papers.keys()),author_papers.keys()))
    vec = CountVectorizer()
    papers_vec = vec.fit_transform(author_papers.values())
    new_author_paper=''
    new_vec=vec.transform(new_author_paper)
    k_max_sim=find_to_authors(papers_vec,new_vec)#作者索引
    k_max_sim_names=list(map(lambda x:idx_to_name(x),k_max_sim))#最高相似作者
