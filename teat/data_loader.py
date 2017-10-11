import pandas as pd
import numpy as np
import time,datetime

def data_loader():
	df=pd.read_csv('1.txt',sep=',')
	date=df['date'].drop_duplicates()

	target_times=range(9,23)

	for da in date:
		# da: someday 
		sub_df=df[df['date']==da]
		all_time=sub_df['start']
		time_avable=list(map(lambda x:judge(x,all_time) ,target_times))

		
		time_avable=[x for x in time_avable if x !=None]	
		week_feature(da)
		for t in time_avable:			
			# time_f=time_feature(df,da,t)
			# three_f=three_hours_feature(sub_df,t)
			week_feature(da)



def judge(tar_time,all_time):
	tar_list=[tar_time,tar_time-1,tar_time-2,tar_time-3]
	if len(set(tar_list).difference(set(all_time)))==0:
		return tar_time




def time_feature(df,da,t):
	print type(da)
	# print df[df['start']==9]
	sub_df_befor=df[np.logical_and(df['date']<=da,df['start']==t)]
	bumber_list=sub_df_befor['number']

	mean_=np.mean(bumber_list)
	var_=np.var(bumber_list)
	mediam_=np.median(bumber_list)
	percentile_25=np.percentile(bumber_list,25)
	percentile_75=np.percentile(bumber_list,75)

	return mean_,var_,mediam_,percentile_25,percentile_75


def three_hours_feature(sub_df,tar_time):
	tar_list=[tar_time,tar_time-1,tar_time-2,tar_time-3]
	num_list=[]
	for h in tar_list:
		sub_data=sub_df[sub_df['start']==h]
		assert sub_data.shape[0]==1
		num_list.append(sub_data['number'].tolist()[0])
		
	print num_list




def week_feature(da):
	da_int_list=[int(x) for x in da.split('-')]
	day = datetime.datetime(*da_int_list).weekday()
	one_hot=np.zeros(7)
	one_hot[day]=1.0
	return one_hot
	

	
	


if __name__ == '__main__':
	data_loader()
