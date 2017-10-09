# -*- coding:utf-8 -*-
import requests
import codecs
from bs4 import BeautifulSoup as bs
import re
import json
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
def geturl(url):
    urls=[]
    try:
        content=requests.get(url,timeout=3).text
    except:
        pass
    content=bs(content,'lxml')
    lianjie=content.findAll('h3',class_='r')
    for a in lianjie:
        urls.append(a.find('a')["href"])
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
    score_list={}
    for _url in list_url:
        score=0
        _url=_url.lower() #域名转换为小写

        if org_domain_name in _url:
            domain_name_existed=True
            score=score+1

        for subname in scholar_subname_list:
            subname=subname.strip('.')
            if subname in _url:
                scholar_name_existed=True
                score=score+0.5
        score_list[_url]=score
    score_list=sorted(score_list.items(),key=lambda d:d[1],reverse=True)

    return score_list

def getxueshu():
    xueshu=[]
    with codecs.open("training.txt","r","utf-8") as f:
        for line in f.readlines():
            if '#name:' in line:
                name=line.split(':')[1].strip('\n')
                xueshu.append(name)
                continue


            if '#search_results_page' in line:
                results_page=line[21:].strip('\n')
                xueshu.append(results_page)
                continue

            if '#homepage' in line:
                homepage=line[10:].strip('\n')
                if line.count('/')>2:
                        yuming=re.compile(r'//(.*?)/').search(line).group(1)
                elif 'http' in line:
                        yuming=line.split('//')[1]
                else:
                        yuming=re.compile(r':(.*?)/').search(line).group(1)
                xueshu.append([homepage,yuming])
                continue

    xueshu=[xueshu[i:i+3] for i in range(0,len(xueshu),3)]
    return xueshu

def zfyb():
    sum=0
    fuyb=[]
    zyb=[]
    xueshu=getxueshu()
    fx=codecs.open("xueshu.json","r","utf-8")
    xueshudict=json.loads(fx.read())
    for a in xueshu:
        if a[2][0] in xueshudict[a[1]]:
            zyb.append(a[2][0])
            xueshudict[a[1]].remove(a[2][0])
            fuyb.extend(xueshudict[a[1]])
            f=open('zyb.txt','a',encoding='utf-8')
            f.write(a[2][0]+'\n')

    with open('fuyb.txt','a',encoding='utf-8') as f:
        for a in fuyb:
            f.write(a+'\n')

def bys(url):
    data = []
    target = []
    dir = ['fuyb.txt','zyb.txt']
    for path in dir:
        with open(path,'r') as f:
            for line in f:
                line = line.strip('\n')
                data.append(line)
                if path == 'zyb.txt':
                    target.append(1)
                else:
                    target.append(0)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=33)
    vec = CountVectorizer()
    X_train = vec.fit_transform(X_train)
    X_test = vec.transform(X_test)
    # mnb = RandomForestClassifier()
    # mnb.fit(X_train, y_train)
    # joblib.dump(mnb, 'bys.model')
    bys = joblib.load('bys.model')
    return bys.predict(vec.transform([url]))
    # print( 'The accuracy is',bys.score(X_test, y_test))

def chulistr(url):
    url=re.sub('\\W',' ',url)
    url=re.sub(' +',' ',url)
    return url

def prezhuye(org,name,seapage):
        # fx=codecs.open("xueshu.json","r","utf-8")
        # xueshudict=json.loads(fx.read())
        fx=codecs.open("org.json","r","utf-8")
        orgdict=json.loads(fx.read())
        urllist=geturl(seapage)
        if org in orgdict:
            org=orgdict[org]
        urls=score_url(org,name,urllist)
        for b in urls[0:6]:
            url=chulistr(b[0])
            if bys(url)==[1]:
                return b[0]
        return urls[0][0]

def validation():
    validation=[]
    with codecs.open("task1_test_final.txt","r","utf-8") as f:
        for line in f.readlines():
            if '#id:' in line:
                id=line[4:].strip('\n')
                validation.append(id)
                continue

            if '#name:' in line:
                name=line[6:].strip('\n')
                validation.append(name)
                continue


            if '#org' in line:
                org=line[5:].strip('\n')
                validation.append(org)
                continue

            if '#search_results_page' in line:
                search_results_page=line[21:].strip('\n')
                validation.append(search_results_page)
                continue

    validation=[validation[i:i+4] for i in range(0,len(validation),4)]
    return validation

# print(prezhuye('清华大学电机工程与应用电子技术系','Ying Chen','http://166.111.7.106:8081/5429a7e9dabfae864af8bd0c.html'))
# print(validation())
def paqu(url):
    content=requests.get(url,timeout=5)
    content.encoding='utf-8'
    content=bs(content.text,'lxml')
    body=content.find('body')
    gender='m'
    position='position'
    photo='photo'
    email='email'
    zhicheng=""
    zhiwei=""
    location=""
    for tag in body.descendants:
        try:
            email=re.search("[A-Za-z0-9\.]+@[\S]*",tag.get_text()).group().strip()
            if email!='email':
                break
        except:
            continue
    for tag in body.descendants:
        try:
            photo=re.search(".*.(jpg|jpeg|png|png)",tag['src']).group()
            if photo!='photo':
                break
        except:
            continue

    for tag in body.descendants:
        try:
            zhicheng=re.search("\S* (讲师|副教授|教授|professor|lecturer|research fellow)",tag.get_text(),re.I).group()
            zhiwei=re.search("(chancellor|president|chair|director|manager)",tag.get_text(),re.I).group()
            if position!='position':
                break
        except:
            continue
    if zhiwei=="":
        position=zhicheng
    else:
        position=zhicheng+';'+zhiwei

    for tag in body.descendants:
        try:
            location=re.search("address:(.*)",tag.get_text(),re.I).group(1).strip()
            if location!="":
                break
        except:
            continue
    return [gender,position,photo,email,location]


num=1
yzj=validation()
# print(yzj)
for author in yzj:
        try:
            if num<=2711:
                num=num+1
                continue
            else:
                print(author[1],str(num))
                url=prezhuye(author[2],author[1],author[3])
                paqulist=paqu(url)
                f=open('task1_final.txt_0','a',encoding='utf-8')
                f.write(author[0]+'    '+url+'    '+'    '.join(paqulist)+'\n')
                num=num+1
            if num==3001:
                print('Done')
                break

        except:
            f=open('task1_final.txt_0','a',encoding='utf-8')
            f.write(author[0]+'    '+'homepage'+'    '.join(['gender','position','photo','email','location'])+'\n')













# fx=codecs.open("xueshu.json","r","utf-8")
# xueshudict=json.loads(fx.read())
# print(score_url('www.cs.toronto.edu','Diane Horton',xueshudict['http://166.111.7.106:8081/53f43cbadabfaedd74dd56dc.html']))
# aaa=score_url('www.cs.toronto.edu','Diane Horton',xueshudict['http://166.111.7.106:8081/53f43cbadabfaedd74dd56dc.html'])

# for a in xueshu:
#     sum=sum+1
#     print(sum)
#     urls=geturl(a[1])
#     prehome=score_url(a[2][1],a[0],urls)
#     if prehome==a[2][0]:
#         score=score+1
#
# print(score/len(xueshu))
# for i in range(0,45,3):
#     print(i)

# for a in xueshu:
#     # requests.adapters.DEFAULT_RETRIES = 5
#     urls=geturl(a[1])
#     fp=open('xueshu.txt','a')
#     fp.write(a[1]+':'+str(urls)+'\n')
#     sum=sum+1
#     print(sum)
# content={}
# fx=open('xueshu.txt','r')
# for line in fx:
#     sum=sum+1
#     print(sum)
#     line=line.split('[')
#     search_results_page=line[0].strip(':')
#     hxpage=line[1].replace(']','')
#     hxpage=hxpage.replace("'",'').split(',')
#     content[search_results_page]=hxpage
#
# fp=open('xueshu.json','w')
# json.dump(content,fp,indent=4)
# print("Done")
# fp.close()
