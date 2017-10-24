import heapq
import numpy as np
import pandas as pd
def data_hlper():
    loc = pd.read_csv('C:\\Users\\5201314nana\\Desktop\\zhao\\shop_location.csv')
    df_user_infos=pd.read_csv('C:\\Users\\5201314nana\\Desktop\\zhao\\sub50w.csv')
    df_shop_infos=pd.read_csv('C:\\Users\\5201314nana\\Desktop\\zhao\\train-ccf_first_round_shop_info.csv')
    #连接全表
    df_usr_data=pd.merge(df_user_infos,df_shop_infos,on=['shop_id'],how='left')
    df_shop_data=pd.merge(loc,df_shop_infos,on=['shop_id'],how='left')

    count=0
    for index, row in df_usr_data.iterrows():
        res_list = get_shop_dis(loc, row['longitude_x'], row['latitude_x'])
        if row['shop_id'] in res_list.tolist():
            count += 1
        if index % 1000==0:
            print (index,count)


def get_shop_dis(loc,lo,la):
    shop_list = loc.icol(0)
    lo=np.array([lo]*len(shop_list))
    la=np.array([la]*len(shop_list))
    usrloc_fill_mat=np.matrix(list(zip(lo,la)))
    loc_mat=np.matrix(list(zip(loc.icol(1),loc.icol(2))))
    diff_mat=(np.power(usrloc_fill_mat-loc_mat,2).sum(axis=1)).tolist()
    diff_idx=heapq.nsmallest(20, diff_mat)

    return (shop_list[[diff_mat.index(x) for x in list(diff_idx)]])




if __name__ == '__main__':
    data_hlper()
