# -*- coding:utf-8 -*-
import requests
import codecs
from bs4 import BeautifulSoup as bs
import re
def geturl(url):
    urls=[]
    content=requests.get(url).text
    content=bs(content,'lxml')
    lianjie=content.findAll('h3')
    for a in lianjie:
        urls.append(a.find('a')['href'])
    return urls

def score_url(org_domain_name='',scholar_name='',list_url=''):
    """
    org_domain_name:组织域名
    scholar_name：学者名
    list_url：待选url集合
    """
    scholar_subname_list=scholar_name.lower().split(' ')
    domain_name_existed=False
    scholar_name_existed=False
    score_list=[]
    for _url in list_url:
        score=0
        _url=_url.lower() #域名转换为小写

        if org_domain_name in _url:
            domain_name_existed=True
            score=score+1

        for subname in scholar_subname_list:
            if subname in _url:
                scholar_name_existed=True
                score=score+1
        score_list.append(score)
    max_index=score_list.index(max(score_list))
    return list_url[max_index]

score=0
sum=0
xueshu=[]
with codecs.open("task1/training.txt","r","utf-8") as f:
    for line in f.readlines():
        if '#name:' in line:
            name=line.split(':')[1].strip('\n')
            # print(name)
            continue


        if '#search_results_page' in line:
            results_page=line[21:]
            if line.count('/')>2:
                    yuming=re.compile(r'//(.*?)/').search(line).group(1)
            elif 'http' in line:
                    yuming=line.split('//')[1]
            else:
                    yuming=re.compile(r':(.*?)/').search(line).group(1)
            # print(results_page)
            continue

        if '#homepage' in line:
            homepage=line[10:]
            continue


            # print(homepage)
    # urls=geturl(results_page)
    # prehome=score_url(yuming,name,urls)
    # if prehome==homepage:
    #     score=score+1
    #
    # print(score/5998)
