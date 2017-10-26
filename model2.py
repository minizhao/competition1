# -*- coding: utf-8 -*-
import heapq
import numpy as np
import pandas as pd
import bottleneck
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import re
wname_2_idx=dict()

def data_hlper(sub_usr_df,sub_shop_df):
    
    count=0
    x_data=[]
    y_data=[]

    for index, row in sub_usr_df.iterrows():
        top20_shop_idx = get_shop_dis(sub_shop_df, row['longitude_x'], row['latitude_x'])
        top5_wifi_name=get_wifi_top5(wifi_str=row['wifi_infos'])
        time_point=get_time_point(row['time_stamp'])
        # category_price=[sub_digt(row['category_id']),row['price']]
        
        x_data.append(np.concatenate([top20_shop_idx,top5_wifi_name,time_point]))     
        y_data.append(row['shop_id'])

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
    loc = pd.read_csv('shop_location.csv')
    df_user_infos=pd.read_csv('sub50w.csv')
    df_shop_infos=pd.read_csv('train-ccf_first_round_shop_info.csv')
    #连接全表
    df_usr_data=pd.merge(df_user_infos,df_shop_infos,on=['shop_id'],how='left')
    df_shop_data=pd.merge(loc,df_shop_infos,on=['shop_id'],how='left')
    
    all_mallid=df_shop_infos['mall_id'].drop_duplicates()#所有的mall_id
    
    for ma_id in all_mallid:
        sub_usr_df=df_usr_data[df_usr_data['mall_id']==ma_id]
        sub_shop_df=df_shop_data[df_shop_data['mall_id']==ma_id]

        x_data,y_data=data_hlper(sub_usr_df,sub_shop_df)
        X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, \
            test_size=0.25, random_state=33)

        print ('start')
        GBC=GradientBoostingClassifier()
        GBC=GBC.fit(X_train, y_train)
        
        print ("GBC of {} train score  is {}".format(ma_id,GBC.score(X_test, y_test)))
        wname_2_idx.clear() 
