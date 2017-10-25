# -*- coding: utf-8 -*-
import heapq
import numpy as np
import pandas as pd
import bottleneck
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import re
wname_2_idx=dict()

def data_hlper():
    loc = pd.read_csv('shop_location.csv')
    df_user_infos=pd.read_csv('sub50w.csv')
    df_shop_infos=pd.read_csv('train-ccf_first_round_shop_info.csv')
    #连接全表
    df_usr_data=pd.merge(df_user_infos,df_shop_infos,on=['shop_id'],how='left')
    df_shop_data=pd.merge(loc,df_shop_infos,on=['shop_id'],how='left')

    count=0
    x_data=[]
    y_data=[]
    for index, row in df_usr_data.iterrows():
        top20_shop_idx = get_shop_dis(loc, row['longitude_x'], row['latitude_x'])
        top5_wifi_name=get_wifi_top5(wifi_str=row['wifi_infos'])
        time_point=get_time_point(row['time_stamp'])
        category_price=[sub_digt(row['category_id']),row['price']]
        
        x_data.append(np.concatenate([top20_shop_idx,top5_wifi_name,time_point,category_price]))     
        y_data.append(row['shop_id'])

        if index % 1000==0:
            print (index)
    assert len(x_data)==len(y_data)
    return x_data,y_data


def get_shop_dis(loc,lo,la):
    shop_list = loc['shop_id']
    lo=np.array([lo]*len(shop_list))
    la=np.array([la]*len(shop_list))
    usrloc_fill_mat=np.matrix(list(zip(lo,la)))
    loc_mat=np.matrix(list(zip(loc['mean_lo'],loc['mean_la'])))
    diff_mat=(np.power(usrloc_fill_mat-loc_mat,2).sum(axis=1))
    diff_mat=diff_mat.reshape(1,len(diff_mat))
    top_20_idx= bottleneck.argpartition(diff_mat, 20)[:20]
    return top_20_idx[0]

def get_wifi_top5(wifi_str=''):
    str_list=wifi_str.split(';')
    wifi_ifos=np.array([x.strip(' ').split('|')[:2] for x in str_list])
    w_name=wifi_ifos[:,0]
    w_value=[int(x) for x in wifi_ifos[:,1]]
    if len(wifi_ifos)>5:
        top_5_idx= bottleneck.argpartition(-np.array(w_value), 5)[:5] 
        return wf_name_2_idx(w_name[top_5_idx])
        
    else:
        sort_idx=np.argsort(-np.array(w_value))
        w_name=w_name[sort_idx].tolist()
        w_name.extend(['b_null']*(5-len(wifi_ifos)))
        return wf_name_2_idx(w_name)

def get_time_point(time_stamp=''):
    rr = re.compile(r'[\d]+:[\d]+')
    match_list=rr.findall(time_stamp)
    time_point=match_list[0].split(':')[0]
    return [time_point]

def sub_digt(str_=''):
    rr = re.compile(r'[\d]+')
    match_list=rr.findall(str_)
    digt=match_list[0]
    return digt


def wf_name_2_idx(w_name):
    w_idx=[]
    for w in w_name:
        if w in wname_2_idx.keys():
            w_idx.append(wname_2_idx[w])
        else:
            wname_2_idx[w]=len(wname_2_idx)
            w_idx.append(wname_2_idx[w])
    return w_idx


if __name__ == '__main__':
    x_data,y_data=data_hlper()
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, \
        test_size=0.25, random_state=33)

    print ('start')
    GBC=GradientBoostingClassifier()
    GBC=GBC.fit(X_train, y_train)

    print ("GBR train score is {}".format(GBC.score(X_test, y_test)))
