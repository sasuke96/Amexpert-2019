
#import libraries

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from datetime import datetime
import datetime
from catboost import Pool, CatBoostClassifier, cv
import pickle as pkl
import matplotlib.pyplot as plt


#load_data

filepath = '/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/'

train = pd.read_csv(filepath + 'train.csv.filepart')
test = pd.read_csv(filepath + 'test_QyjYwdj.csv.filepart')
trans = pd.read_csv(filepath + 'customer_transaction_data.csv.filepart')
item_data = pd.read_csv(filepath + 'item_data.csv.filepart')
campaign_data = pd.read_csv(filepath + 'campaign_data.csv')
coupon_item_mapping = pd.read_csv(filepath + 'coupon_item_mapping.csv.filepart')
customer_demographics = pd.read_csv(filepath + 'customer_demographics.csv')
sample_submission = pd.read_csv(filepath + 'sample_submission_Byiv0dS.csv.filepart')



#phase 1
campaign_data["start_date"]=pd.to_datetime(campaign_data["start_date"],dayfirst=True)
campaign_data["end_date"]=pd.to_datetime(campaign_data["end_date"],dayfirst=True)
train_cmp = pd.merge(train , campaign_data, on='campaign_id', how = 'left')
test_cmp = pd.merge(test , campaign_data, on='campaign_id', how = 'left')


train_cmp.sort_values(by=['start_date'], inplace=True)
test_cmp.sort_values(by=['start_date'], inplace=True)

data = train_cmp.append(test_cmp, sort = False)

data_item =pd.merge(pd.merge(data,coupon_item_mapping,how='left', on = 'coupon_id'), item_data,on= 'item_id', how ='left')
data_item.head()

data_merged =pd.merge(data_item, customer_demographics, on ='customer_id', how ='left')

data_merged.age_range.isna().sum()  #40% missing customer info

#creating feature based on target variable
ATTRIBUTION_CATEGORIES = [        
    ['customer_id'],
    ['campaign_type'],
    ['coupon_id'],
]

#calculating confRate for merged train + test data  
freqs = {}
for cols in ATTRIBUTION_CATEGORIES:
    
    # New feature name

    new_feature = '_'.join(cols)+'_confRate'    
    
    # Perform the groupby
    group_object = data_merged.groupby(cols)
    
    # Group sizes    
    group_sizes = group_object.size()
    log_group = np.log(100000) # 1000 views -> 60% confidence, 100 views -> 40% confidence 
    print(">> Calculating confidence-weighted rate for: {}.\n   Saving to: {}. Group Max /Mean / Median / Min: {} / {} / {} / {}".format(
        cols, new_feature, 
        group_sizes.max(), 
        np.round(group_sizes.mean(), 2),
        np.round(group_sizes.median(), 2),
        group_sizes.min()
    ))
    
    # Aggregation function
    def rate_calculation(x):
        """Calculate the attributed rate. Scale by confidence"""
        rate = x.sum() / float(x.count())
        conf = np.min([1, np.log(x.count()) / log_group])
        return rate * conf
    
    # Perform the merge
    data_merged = data_merged.merge(
        group_object['redemption_status']. \
            apply(rate_calculation). \
            reset_index(). \
            rename( 
                index=str,
                columns={'redemption_status': new_feature}
            )[cols + [new_feature]],
        on=cols, how='left'
    )

data_merged.head()
# mapping features to test data
# for cols in ATTRIBUTION_CATEGORIES2:
#     temp = train2.set_index(cols)[cols+'_confRate'].to_dict()
#     test2[cols+'_confRate']= test2[cols].map(temp) 
#     print(cols)


data =data_merged.copy()

data['cmpg_cust_diff'] = data['campaign_id'].map(data.groupby('campaign_id')['customer_id'].apply(lambda x: len(x) - x.unique().size).to_dict())
data['cmpg_cust_ratio'] = data['campaign_id'].map(data.groupby('campaign_id')['customer_id'].apply(lambda x: x.unique().size / len(x)).to_dict())

data['cmpg_cpn_diff'] = data['campaign_id'].map(data.groupby('campaign_id')['coupon_id'].apply(lambda x: len(x) - x.unique().size).to_dict())
data['cmpg_cpn_ratio'] = data['campaign_id'].map(data.groupby('campaign_id')['coupon_id'].apply(lambda x: x.unique().size / len(x)).to_dict())

data['cmpg_item_diff'] = data['campaign_id'].map(data.groupby('campaign_id')['item_id'].apply(lambda x: len(x) - x.unique().size).to_dict())
data['cmpg_item_ratio'] = data['campaign_id'].map(data.groupby('campaign_id')['item_id'].apply(lambda x: x.unique().size / len(x)).to_dict())

data['cust_cmpg_diff'] = data['customer_id'].map(data.groupby('customer_id')['campaign_id'].apply(lambda x: len(x) - x.unique().size).to_dict())
data['cust_cmpg_ratio'] = data['customer_id'].map(data.groupby('customer_id')['campaign_id'].apply(lambda x: x.unique().size / len(x)).to_dict())

data['cust_cmpgt_diff'] = data['customer_id'].map(data.groupby('customer_id')['campaign_type'].apply(lambda x: len(x) - x.unique().size).to_dict())
data['cust_cmpgt_ratio'] = data['customer_id'].map(data.groupby('customer_id')['campaign_type'].apply(lambda x: x.unique().size / len(x)).to_dict())

data['cust_cpn_diff'] = data['customer_id'].map(data.groupby('customer_id')['coupon_id'].apply(lambda x: len(x) - x.unique().size).to_dict())
data['cust_cpn_ratio'] = data['customer_id'].map(data.groupby('customer_id')['coupon_id'].apply(lambda x: x.unique().size / len(x)).to_dict())

data['cust_item_diff'] = data['customer_id'].map(data.groupby('customer_id')['coupon_id'].apply(lambda x: len(x) - x.unique().size).to_dict())
data['cust_item_ratio'] = data['customer_id'].map(data.groupby('customer_id')['coupon_id'].apply(lambda x: x.unique().size / len(x)).to_dict())

data['cpn_cmpgt_diff'] = data['coupon_id'].map(data.groupby('coupon_id')['campaign_type'].apply(lambda x: len(x) - x.unique().size).to_dict())
data['cpn_cmpgt_ratio'] = data['coupon_id'].map(data.groupby('coupon_id')['campaign_type'].apply(lambda x: x.unique().size / len(x)).to_dict())


demo_cols = ['age_range', 'marital_status', 'rented', 'family_size', 'no_of_children', 'income_bracket']

for col in demo_cols:
    data[col].fillna(data[col].mode()[0],inplace=True)
	
	
data_nodup =  data.drop_duplicates(['customer_id','campaign_id','coupon_id','redemption_status'])

data_nodup.to_pickle('/axp/buanalytics/csgrbc/dev/Divya/AmExpert/data_nodup_features.pkl')

data_nodup2 =pkl.load(open('/axp/buanalytics/csgrbc/dev/Divya/AmExpert/data_nodup_features.pkl','rb'))


#phase 2

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from datetime import datetime
import datetime
from catboost import Pool, CatBoostClassifier, cv

train=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/train.csv.filepart")
test=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/test_QyjYwdj.csv.filepart")
sample_submission=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/sample_submission_Byiv0dS.csv.filepart")
item_data=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/item_data.csv.filepart")
ct_data=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/customer_transaction_data.csv.filepart")
ct_demographic=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/customer_demographics.csv")
coupon_item_map=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/coupon_item_mapping.csv.filepart")
campaign_data=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/campaign_data.csv")

coupoun_details=pd.merge(coupon_item_map,item_data,how="left",on="item_id")
coupon_final=coupoun_details[['coupon_id']].drop_duplicates()

cols=['item_id','brand','brand_type','category']

for col in cols:
    group=coupoun_details.groupby(['coupon_id']).agg({col:['nunique']})
    group.columns=['distinct_'+col]
    group.reset_index(inplace=True)
    coupon_final=pd.merge(coupon_final,group,how="left",on="coupon_id")

for col in cols:
    group=coupoun_details.groupby(['coupon_id']).agg({col:['count']})
    group.columns=['count_'+col]
    group.reset_index(inplace=True)
    coupon_final=pd.merge(coupon_final,group,how="left",on="coupon_id")
	
for col in cols:
    coupon_final[col+'_ratio']=coupon_final['distinct_'+col]/coupon_final['count_'+col]

for col in cols:
    group=coupoun_details.groupby(['coupon_id'])[col].value_counts().reset_index(name='mode_vc_'+col)
    group.drop_duplicates(['coupon_id'],keep='first',inplace=True)
    coupon_final=pd.merge(coupon_final,group,how="left",on="coupon_id")

data=pd.concat([train,test])
data.shape

data=pd.merge(data,campaign_data,how='left',on='campaign_id')
data=pd.merge(data,ct_demographic,how='left',on='customer_id')
data.shape

cols=['age_range','marital_status', 'rented', 'family_size', 'no_of_children',
       'income_bracket']
for col in cols:
    data["null_"+col]=np.where(data[col].isnull()==True,1,0)

data=pd.merge(data,coupon_final,how='left',on='coupon_id')

data["start_date"]=pd.to_datetime(data["start_date"],dayfirst=True)

data["end_date"]=pd.to_datetime(data["end_date"],dayfirst=True)

data["campaign_length"]=(data["end_date"]-data["start_date"])/ np.timedelta64(1, 'D')

cols=['age_range','marital_status', 'rented', 'family_size', 'no_of_children',
       'income_bracket']

for col in cols:
    data[col].fillna(data[col].mode()[0],inplace=True)
	
filename = "/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/int_data/attempt2"
outfile = open(filename,'wb')
pickle.dump(data,outfile)
outfile.close()


#feature based on the transaction less than end date

ct_data['other_discount_percentage']=(abs(ct_data['other_discount'])/(ct_data['selling_price']-ct_data['other_discount']-ct_data['other_discount']))*100
ct_data['coupon_discount_percentage']=(abs(ct_data['coupon_discount'])/(ct_data['selling_price']-ct_data['other_discount']-ct_data['other_discount']))*100

ct_data['date']=pd.to_datetime(ct_data['date'],yearfirst=True)
new_data=pd.merge(ct_data,item_data,how="left",on="item_id")

data=pd.concat([train,test])
data=pd.merge(data,campaign_data,how='left',on='campaign_id')

data2=data[['customer_id','campaign_type','campaign_id','start_date', 'end_date']]

data2.drop_duplicates(inplace=True)

ct_data['other_discount_percentage']=(abs(ct_data['other_discount'])/(ct_data['selling_price']-ct_data['other_discount']-ct_data['other_discount']))*100
ct_data['coupon_discount_percentage']=(abs(ct_data['coupon_discount'])/(ct_data['selling_price']-ct_data['other_discount']-ct_data['other_discount']))*100

data2["start_date"]=pd.to_datetime(data2["start_date"],dayfirst=True)

data2["end_date"]=pd.to_datetime(data2["end_date"],dayfirst=True)

ct_new_data2=ct_new_data[ct_new_data.date<ct_new_data.end_date]

ct_group=ct_new_data2.groupby(['customer_id','campaign_id']).agg({'item_id':'count'})
ct_group.columns=['num_item_id']
ct_group.reset_index(inplace=True)

group=ct_new_data2.groupby(['customer_id','campaign_id']).agg({'item_id':'nunique'})
group.columns=['unique_item_id']
group.reset_index(inplace=True)
ct_group=pd.merge(ct_group,group,how="left",on=['customer_id','campaign_id'])
print (ct_group.shape)

ct_group['item_ratio']=ct_group['unique_item_id']/ct_group['num_item_id']
ct_group['item_diff']=ct_group['num_item_id']-ct_group['unique_item_id']

cols=[ 'quantity', 'selling_price','other_discount_percentage',
       'coupon_discount_percentage']
for col in cols:
    group=ct_new_data2.groupby(['customer_id','campaign_id']).agg({col:'mean'})
    group.columns=['avg_'+col]
    group.reset_index(inplace=True)
    ct_group=pd.merge(ct_group,group,how="left",on=['customer_id','campaign_id'])


cols=['brand', 'brand_type', 'category']

for col in cols:
    group=ct_new_data2.groupby(['customer_id','campaign_id'])[col].apply(lambda x: x.value_counts().index[0]).reset_index()
    group.columns=['customer_id','campaign_id','mode_'+col]
    ct_group=pd.merge(ct_group,group,how="left",on=['customer_id','campaign_id'])

group=ct_new_data2.groupby(['customer_id','campaign_id','date']).agg({'item_id':'nunique','quantity':'sum','selling_price':'sum','other_discount_percentage':'mean','coupon_discount_percentage':'mean'})
group.reset_index(inplace=True)

group.sort_values(by=['customer_id', 'campaign_id','date'],inplace=True)

group['day_order_diff'] = (group.groupby(['customer_id','campaign_id'])['date'].diff())/np.timedelta64(1, 'D')

group=pd.merge(group,campaign_data[['campaign_id','end_date']],how='left',on='campaign_id')

group["end_date"]=pd.to_datetime(group["end_date"],dayfirst=True)

group['diff_end_date']=(group['end_date']-group['date'])/np.timedelta64(1, 'D')

random=['max','min','mean']
for ran in random:
    group2=group.groupby(['customer_id','campaign_id']).agg({'diff_end_date':ran})
    group2.columns=['diff_end_date_'+ran]
    group2.reset_index(inplace=True)
    ct_group=pd.merge(ct_group,group2,how='left',on=['customer_id','campaign_id'])
    
filename = "/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/int_data/ct_feature_end_date"
outfile = open(filename,'wb')
pickle.dump(ct_group,outfile)
outfile.close()


#phase 3
train=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/train.csv.filepart")
test=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/test_QyjYwdj.csv.filepart")
sample_submission=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/sample_submission_Byiv0dS.csv.filepart")
item_data=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/item_data.csv.filepart")
ct_data=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/customer_transaction_data.csv.filepart")
ct_demographic=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/customer_demographics.csv")
coupon_item_map=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/coupon_item_mapping.csv.filepart")
campaign_data=pd.read_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/campaign_data.csv")

ct_data['other_discount_percentage']=(abs(ct_data['other_discount'])/(ct_data['selling_price']-ct_data['other_discount']-ct_data['other_discount']))*100
ct_data['coupon_discount_percentage']=(abs(ct_data['coupon_discount'])/(ct_data['selling_price']-ct_data['other_discount']-ct_data['other_discount']))*100

ct_data['date']=pd.to_datetime(ct_data['date'],yearfirst=True)
new_data=pd.merge(ct_data,item_data,how="left",on="item_id")

data=pd.concat([train,test])
data=pd.merge(data,campaign_data,how='left',on='campaign_id')

data2=data[['customer_id','campaign_type','campaign_id','start_date', 'end_date']]

data2.drop_duplicates(inplace=True)

data2["start_date"]=pd.to_datetime(data2["start_date"],dayfirst=True)

data2["end_date"]=pd.to_datetime(data2["end_date"],dayfirst=True)

ct_data['other_discount_percentage']=(abs(ct_data['other_discount'])/(ct_data['selling_price']-ct_data['other_discount']-ct_data['other_discount']))*100
ct_data['coupon_discount_percentage']=(abs(ct_data['coupon_discount'])/(ct_data['selling_price']-ct_data['other_discount']-ct_data['other_discount']))*100

ct_data['date']=pd.to_datetime(ct_data['date'],yearfirst=True)

new_data=pd.merge(ct_data,item_data,how="left",on="item_id")

ct_new_data=pd.merge(new_data,data2,how='inner',on='customer_id')

ct_new_data3=ct_new_data[(ct_new_data.start_date<=ct_new_data.date)&(ct_new_data.date<=ct_new_data.end_date)] #data between end date and start date of campaign

ct_new_data4=ct_new_data[ct_new_data.date<ct_new_data.start_date] #data before start_date

ct_new_data2=ct_new_data[ct_new_data.date<ct_new_data.end_date] #more feature for end date

ct_cmp_id=ct_new_data2[['customer_id','campaign_id']].drop_duplicates()  #feature  end date concept on customer_id and campaign_id

ct_cmp_type=ct_new_data2[['customer_id','campaign_type']].drop_duplicates()  #feature end date concept on customer_id and campaign_type

ct_cmp_type2=ct_new_data3[['customer_id','campaign_type']].drop_duplicates()  #features for date between end_date and start_date

ct_cmp_type3=ct_new_data4[['customer_id','campaign_type']].drop_duplicates()  #feature for date before start date


#feature  end date concept on customer_id and campaign_id

group=ct_new_data2.groupby(['customer_id','campaign_id']).agg({'brand':'nunique'})
group.columns=['unique_brand_id']
group.reset_index(inplace=True)
ct_cmp_id=pd.merge(ct_cmp_id,group,how="left",on=['customer_id','campaign_id'])


group=ct_new_data2.groupby(['customer_id','campaign_id']).agg({'brand':'count'})
group.columns=['count_brand_id']
group.reset_index(inplace=True)
ct_cmp_id=pd.merge(ct_cmp_id,group,how="left",on=['customer_id','campaign_id'])

ct_cmp_id['brand_ratio']=ct_cmp_id['unique_brand_id']/ct_cmp_id['count_brand_id']
ct_cmp_id['brand_diff']=ct_cmp_id['count_brand_id']-ct_cmp_id['unique_brand_id']

cols=['brand_type', 'category']

for col in cols:
    group=ct_new_data2.groupby(['customer_id','campaign_id']).agg({col:'nunique'})
    group.columns=['unique_'+col]
    group.reset_index(inplace=True)
    ct_cmp_id=pd.merge(ct_cmp_id,group,how="left",on=['customer_id','campaign_id'])

cols=['brand', 'brand_type', 'category']

for col in cols:
    group=ct_new_data2.groupby(['customer_id','campaign_id'])[col].value_counts().reset_index(name='mode_vc_'+col)[['customer_id','campaign_id','mode_vc_'+col]]
    group.drop_duplicates(['customer_id','campaign_id'],keep='first',inplace=True)
    ct_cmp_id=pd.merge(ct_cmp_id,group,how="left",on=['customer_id','campaign_id'])

group=ct_new_data2[ct_new_data2.coupon_discount<0].groupby(['customer_id','campaign_id']).agg({'coupon_discount':'count'})
group.columns=['num_coupon_discount_applied']
group.reset_index(inplace=True)
ct_cmp_id=pd.merge(ct_cmp_id,group,how="left",on=['customer_id','campaign_id'])

group=ct_new_data2[ct_new_data2.other_discount<0].groupby(['customer_id','campaign_id']).agg({'other_discount':'count'})
group.columns=['num_other_discount_applied']
group.reset_index(inplace=True)
ct_cmp_id=pd.merge(ct_cmp_id,group,how="left",on=['customer_id','campaign_id'])

ct_cmp_id['ratio_num']=ct_cmp_id['num_coupon_discount_applied']/ct_cmp_id['num_other_discount_applied']

ct_cmp_id.fillna(0,inplace=True)

group=ct_new_data2.groupby(['customer_id','campaign_id']).agg({'selling_price':'max'})
group.columns=['max_'+'selling_price']
group.reset_index(inplace=True)
ct_cmp_id=pd.merge(ct_cmp_id,group,how="left",on=['customer_id','campaign_id'])

group=ct_new_data2.groupby(['customer_id','campaign_id']).agg({'selling_price':'min'})
group.columns=['min_'+'selling_price']
group.reset_index(inplace=True)
ct_cmp_id=pd.merge(ct_cmp_id,group,how="left",on=['customer_id','campaign_id'])

filename = "/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/int_data/new_version_feature"
outfile = open(filename,'wb')
pickle.dump(ct_cmp_id,outfile)
outfile.close()



#feature end date concept on customer_id and campaign_type
ct_cmp_type=ct_new_data2.groupby(['customer_id','campaign_type']).agg({'item_id':'count'})
ct_cmp_type.columns=['num_item_id']
ct_cmp_type.reset_index(inplace=True)

group=ct_new_data2.groupby(['customer_id','campaign_type']).agg({'item_id':'nunique'})
group.columns=['unique_item_id']
group.reset_index(inplace=True)
ct_cmp_type=pd.merge(ct_cmp_type,group,how="left",on=['customer_id','campaign_type'])
print (ct_cmp_type.shape)

ct_cmp_type['item_ratio']=ct_cmp_type['unique_item_id']/ct_cmp_type['num_item_id']
ct_cmp_type['item_diff']=ct_cmp_type['num_item_id']-ct_cmp_type['unique_item_id']

group=ct_new_data2[ct_new_data2.coupon_discount<0].groupby(['customer_id','campaign_type']).agg({'coupon_discount':'count'})
group.columns=['num_coupon_discount_applied']
group.reset_index(inplace=True)
ct_cmp_type=pd.merge(ct_cmp_type,group,how="left",on=['customer_id','campaign_type'])

group=ct_new_data2[ct_new_data2.other_discount<0].groupby(['customer_id','campaign_type']).agg({'other_discount':'count'})
group.columns=['num_other_discount_applied']
group.reset_index(inplace=True)
ct_cmp_type=pd.merge(ct_cmp_type,group,how="left",on=['customer_id','campaign_type'])

ct_cmp_type['ratio_num']=ct_cmp_type['num_coupon_discount_applied']/ct_cmp_type['num_other_discount_applied']

ct_cmp_type.fillna(0,inplace=True)

ct_cmp_type.isnull().sum()

cols=['brand', 'brand_type', 'category']

for col in cols:
    group=ct_new_data2.groupby(['customer_id','campaign_type'])[col].value_counts().reset_index(name='mode_vc_'+col)
    group.drop_duplicates(['customer_id','campaign_type'],keep='first',inplace=True)
    ct_cmp_type=pd.merge(ct_cmp_type,group,how="left",on=['customer_id','campaign_type'])
	

group=ct_new_data2.groupby(['customer_id','campaign_type']).agg({'brand':'nunique'})
group.columns=['unique_brand_id']
group.reset_index(inplace=True)
ct_cmp_type=pd.merge(ct_cmp_type,group,how="left",on=['customer_id','campaign_type'])

group=ct_new_data2.groupby(['customer_id','campaign_type']).agg({'brand':'count'})
group.columns=['count_brand_id']
group.reset_index(inplace=True)
ct_cmp_type=pd.merge(ct_cmp_type,group,how="left",on=['customer_id','campaign_type'])

ct_cmp_type['brand_ratio']=ct_cmp_type['unique_brand_id']/ct_cmp_type['count_brand_id']
ct_cmp_type['brand_diff']=ct_cmp_type['count_brand_id']-ct_cmp_type['unique_brand_id']

cols=[ 'quantity', 'selling_price','other_discount_percentage',
       'coupon_discount_percentage']
for col in cols:
    group=ct_new_data2.groupby(['customer_id','campaign_type']).agg({col:'mean'})
    group.columns=['avg_'+col]
    group.reset_index(inplace=True)
    ct_cmp_type=pd.merge(ct_cmp_type,group,how="left",on=['customer_id','campaign_type'])

group=ct_new_data2.groupby(['customer_id','campaign_type','date']).agg({'item_id':'nunique','quantity':'sum','selling_price':'sum','other_discount_percentage':'mean','coupon_discount_percentage':'mean'})
group.reset_index(inplace=True)

group.sort_values(by=['customer_id', 'campaign_type','date'],inplace=True)
group['day_order_diff'] = (group.groupby(['customer_id','campaign_type'])['date'].diff())/np.timedelta64(1, 'D')

cols= ['item_id', 'other_discount_percentage',
       'selling_price', 'coupon_discount_percentage', 'quantity',
       'day_order_diff']


for col in cols:
    group2=group.groupby(['customer_id','campaign_type']).agg({col:'mean'})
    group2.columns=['day_avg_'+col]
    group2.reset_index(inplace=True)
    ct_cmp_type=pd.merge(ct_cmp_type,group2,how='left',on=['customer_id','campaign_type'])

ct_cmp_type['day_avg_discount_ratio']=ct_cmp_type['day_avg_coupon_discount_percentage']/ct_cmp_type['day_avg_other_discount_percentage']






#feature for date before start date
ct_cmp_type3=ct_new_data4.groupby(['customer_id','campaign_type']).agg({'item_id':'count'})
ct_cmp_type3.columns=['num_item_id']
ct_cmp_type3.reset_index(inplace=True)

group=ct_new_data4.groupby(['customer_id','campaign_type']).agg({'item_id':'nunique'})
group.columns=['unique_item_id']
group.reset_index(inplace=True)
ct_cmp_type3=pd.merge(ct_cmp_type3,group,how="left",on=['customer_id','campaign_type'])
print (ct_cmp_type3.shape)

ct_cmp_type3['item_ratio']=ct_cmp_type3['unique_item_id']/ct_cmp_type3['num_item_id']
ct_cmp_type3['item_diff']=ct_cmp_type3['num_item_id']-ct_cmp_type3['unique_item_id']

group=ct_new_data4[ct_new_data4.coupon_discount<0].groupby(['customer_id','campaign_type']).agg({'coupon_discount':'count'})
group.columns=['num_coupon_discount_applied']
group.reset_index(inplace=True)
ct_cmp_type3=pd.merge(ct_cmp_type3,group,how="left",on=['customer_id','campaign_type'])

group=ct_new_data4[ct_new_data4.other_discount<0].groupby(['customer_id','campaign_type']).agg({'other_discount':'count'})
group.columns=['num_other_discount_applied']
group.reset_index(inplace=True)
ct_cmp_type3=pd.merge(ct_cmp_type3,group,how="left",on=['customer_id','campaign_type'])

ct_cmp_type3['ratio_num']=ct_cmp_type3['num_coupon_discount_applied']/ct_cmp_type3['num_other_discount_applied']

ct_cmp_type3.fillna(0,inplace=True)

ct_cmp_type3.isnull().sum()

cols=['brand', 'brand_type', 'category']

for col in cols:
    group=ct_new_data4.groupby(['customer_id','campaign_type'])[col].value_counts().reset_index(name='mode_vc_'+col)
    group.drop_duplicates(['customer_id','campaign_type'],keep='first',inplace=True)
    ct_cmp_type3=pd.merge(ct_cmp_type3,group,how="left",on=['customer_id','campaign_type'])


group=ct_new_data4.groupby(['customer_id','campaign_type']).agg({'brand':'nunique'})
group.columns=['unique_brand_id']
group.reset_index(inplace=True)
ct_cmp_type3=pd.merge(ct_cmp_type3,group,how="left",on=['customer_id','campaign_type'])

group=ct_new_data4.groupby(['customer_id','campaign_type']).agg({'brand':'count'})
group.columns=['count_brand_id']
group.reset_index(inplace=True)
ct_cmp_type3=pd.merge(ct_cmp_type3,group,how="left",on=['customer_id','campaign_type'])

ct_cmp_type3['brand_ratio']=ct_cmp_type3['unique_brand_id']/ct_cmp_type3['count_brand_id']
ct_cmp_type3['brand_diff']=ct_cmp_type3['count_brand_id']-ct_cmp_type3['unique_brand_id']

cols=[ 'quantity', 'selling_price','other_discount_percentage',
       'coupon_discount_percentage']
for col in cols:
    group=ct_new_data4.groupby(['customer_id','campaign_type']).agg({col:'mean'})
    group.columns=['avg_'+col]
    group.reset_index(inplace=True)
    ct_cmp_type3=pd.merge(ct_cmp_type3,group,how="left",on=['customer_id','campaign_type'])

group=ct_new_data4.groupby(['customer_id','campaign_type','date']).agg({'item_id':'nunique','quantity':'sum','selling_price':'sum','other_discount_percentage':'mean','coupon_discount_percentage':'mean'})
group.reset_index(inplace=True)

group.sort_values(by=['customer_id', 'campaign_type','date'],inplace=True)
group['day_order_diff'] = (group.groupby(['customer_id','campaign_type'])['date'].diff())/np.timedelta64(1, 'D')

cols= ['item_id', 'other_discount_percentage',
       'selling_price', 'coupon_discount_percentage', 'quantity',
       'day_order_diff']


for col in cols:
    group2=group.groupby(['customer_id','campaign_type']).agg({col:'mean'})
    group2.columns=['day_avg_'+col]
    group2.reset_index(inplace=True)
    ct_cmp_type3=pd.merge(ct_cmp_type3,group2,how='left',on=['customer_id','campaign_type'])

ct_cmp_type3['day_avg_discount_ratio']=ct_cmp_type3['day_avg_coupon_discount_percentage']/ct_cmp_type3['day_avg_other_discount_percentage']


ct_cmp_type3.columns=['customer_id',
 'campaign_type',
 'campaign3_num_item_id',
 'campaign3_unique_item_id',
 'campaign3_item_ratio',
 'campaign3_item_diff',
 'campaign3_num_coupon_discount_applied',
 'campaign3_num_other_discount_applied',
 'campaign3_ratio_num',
 'campaign3_brand',
 'campaign3_mode_vc_brand',
 'campaign3_brand_type',
 'campaign3_mode_vc_brand_type',
 'campaign3_category',
 'campaign3_mode_vc_category',
 'campaign3_unique_brand_id',
 'campaign3_count_brand_id',
 'campaign3_brand_ratio',
 'campaign3_brand_diff',
 'campaign3_avg_quantity',
 'campaign3_avg_selling_price',
 'campaign3_avg_other_discount_percentage',
 'campaign3_avg_coupon_discount_percentage',
 'campaign3_day_avg_item_id',
 'campaign3_day_avg_other_discount_percentage',
 'campaign3_day_avg_selling_price',
 'campaign3_day_avg_coupon_discount_percentage',
 'campaign3_day_avg_quantity',
 'campaign3_day_avg_day_order_diff',
 'campaign3_day_avg_discount_ratio']
 
 
 
 
 
 
 
 
 ##features for date between end_date and start_date
 
 ct_cmp_type2=ct_new_data3.groupby(['customer_id','campaign_type']).agg({'item_id':'count'})
ct_cmp_type2.columns=['num_item_id']
ct_cmp_type2.reset_index(inplace=True)

group=ct_new_data3.groupby(['customer_id','campaign_type']).agg({'item_id':'nunique'})
group.columns=['unique_item_id']
group.reset_index(inplace=True)
ct_cmp_type2=pd.merge(ct_cmp_type2,group,how="left",on=['customer_id','campaign_type'])
print (ct_cmp_type2.shape)

ct_cmp_type2['item_ratio']=ct_cmp_type2['unique_item_id']/ct_cmp_type2['num_item_id']
ct_cmp_type2['item_diff']=ct_cmp_type2['num_item_id']-ct_cmp_type2['unique_item_id']

group=ct_new_data3[ct_new_data3.coupon_discount<0].groupby(['customer_id','campaign_type']).agg({'coupon_discount':'count'})
group.columns=['num_coupon_discount_applied']
group.reset_index(inplace=True)
ct_cmp_type2=pd.merge(ct_cmp_type2,group,how="left",on=['customer_id','campaign_type'])

group=ct_new_data3[ct_new_data3.other_discount<0].groupby(['customer_id','campaign_type']).agg({'other_discount':'count'})
group.columns=['num_other_discount_applied']
group.reset_index(inplace=True)
ct_cmp_type2=pd.merge(ct_cmp_type2,group,how="left",on=['customer_id','campaign_type'])

ct_cmp_type2['ratio_num']=ct_cmp_type2['num_coupon_discount_applied']/ct_cmp_type2['num_other_discount_applied']

ct_cmp_type2.fillna(0,inplace=True)

ct_cmp_type2.isnull().sum()

cols=['brand', 'brand_type', 'category']

for col in cols:
    group=ct_new_data3.groupby(['customer_id','campaign_type'])[col].value_counts().reset_index(name='mode_vc_'+col)
    group.drop_duplicates(['customer_id','campaign_type'],keep='first',inplace=True)
    ct_cmp_type2=pd.merge(ct_cmp_type2,group,how="left",on=['customer_id','campaign_type'])


group=ct_new_data3.groupby(['customer_id','campaign_type']).agg({'brand':'nunique'})
group.columns=['unique_brand_id']
group.reset_index(inplace=True)
ct_cmp_type2=pd.merge(ct_cmp_type2,group,how="left",on=['customer_id','campaign_type'])

group=ct_new_data3.groupby(['customer_id','campaign_type']).agg({'brand':'count'})
group.columns=['count_brand_id']
group.reset_index(inplace=True)
ct_cmp_type2=pd.merge(ct_cmp_type2,group,how="left",on=['customer_id','campaign_type'])

ct_cmp_type2['brand_ratio']=ct_cmp_type2['unique_brand_id']/ct_cmp_type2['count_brand_id']
ct_cmp_type2['brand_diff']=ct_cmp_type2['count_brand_id']-ct_cmp_type2['unique_brand_id']

cols=[ 'quantity', 'selling_price','other_discount_percentage',
       'coupon_discount_percentage']
for col in cols:
    group=ct_new_data3.groupby(['customer_id','campaign_type']).agg({col:'mean'})
    group.columns=['avg_'+col]
    group.reset_index(inplace=True)
    ct_cmp_type2=pd.merge(ct_cmp_type2,group,how="left",on=['customer_id','campaign_type'])

group=ct_new_data3.groupby(['customer_id','campaign_type','date']).agg({'item_id':'nunique','quantity':'sum','selling_price':'sum','other_discount_percentage':'mean','coupon_discount_percentage':'mean'})
group.reset_index(inplace=True)

group.sort_values(by=['customer_id', 'campaign_type','date'],inplace=True)
group['day_order_diff'] = (group.groupby(['customer_id','campaign_type'])['date'].diff())/np.timedelta64(1, 'D')

cols= ['item_id', 'other_discount_percentage',
       'selling_price', 'coupon_discount_percentage', 'quantity',
       'day_order_diff']


for col in cols:
    group2=group.groupby(['customer_id','campaign_type']).agg({col:'mean'})
    group2.columns=['day_avg_'+col]
    group2.reset_index(inplace=True)
    ct_cmp_type2=pd.merge(ct_cmp_type2,group2,how='left',on=['customer_id','campaign_type'])

ct_cmp_type2['day_avg_discount_ratio']=ct_cmp_type2['day_avg_coupon_discount_percentage']/ct_cmp_type2['day_avg_other_discount_percentage']

ct_cmp_type2.columns=['customer_id',
 'campaign_type',
 'campaign2_num_item_id',
 'campaign2_unique_item_id',
 'campaign2_item_ratio',
 'campaign2_item_diff',
 'campaign2_num_coupon_discount_applied',
 'campaign2_num_other_discount_applied',
 'campaign2_ratio_num',
 'campaign2_brand',
 'campaign2_mode_vc_brand',
 'campaign2_brand_type',
 'campaign2_mode_vc_brand_type',
 'campaign2_category',
 'campaign2_mode_vc_category',
 'campaign2_unique_brand_id',
 'campaign2_count_brand_id',
 'campaign2_brand_ratio',
 'campaign2_brand_diff',
 'campaign2_avg_quantity',
 'campaign2_avg_selling_price',
 'campaign2_avg_other_discount_percentage',
 'campaign2_avg_coupon_discount_percentage',
 'campaign2_day_avg_item_id',
 'campaign2_day_avg_other_discount_percentage',
 'campaign2_day_avg_selling_price',
 'campaign2_day_avg_coupon_discount_percentage',
 'campaign2_day_avg_quantity',
 'campaign2_day_avg_day_order_diff',
 'campaign2_day_avg_discount_ratio']
 
 
 
random=pd.merge(ct_cmp_type3,ct_cmp_type2,on=['customer_id',
 'campaign_type'],how='left')
 
 random.fillna(0,inplace=True)
 
 cols=['num_item_id', 'unique_item_id','num_coupon_discount_applied',
       'num_other_discount_applied','unique_brand_id', 'count_brand_id','day_avg_item_id','day_avg_quantity',
       'day_avg_day_order_diff']

for col in cols:
    random['common_ratio'+col]=random['campaign2_'+col]/random['campaign3_'+col]

random.fillna(0,inplace=True)

cols=['campaign3_num_item_id',
       'campaign3_unique_item_id', 'campaign3_item_ratio',
       'campaign3_item_diff', 'campaign3_num_coupon_discount_applied',
       'campaign3_num_other_discount_applied', 'campaign3_ratio_num',
       'campaign3_brand', 'campaign3_mode_vc_brand', 'campaign3_brand_type',
       'campaign3_mode_vc_brand_type', 'campaign3_category',
       'campaign3_mode_vc_category', 'campaign3_unique_brand_id',
       'campaign3_count_brand_id', 'campaign3_brand_ratio',
       'campaign3_brand_diff', 'campaign3_avg_quantity',
       'campaign3_avg_selling_price',
       'campaign3_avg_other_discount_percentage',
       'campaign3_avg_coupon_discount_percentage', 'campaign3_day_avg_item_id',
       'campaign3_day_avg_other_discount_percentage',
       'campaign3_day_avg_selling_price',
       'campaign3_day_avg_coupon_discount_percentage',
       'campaign3_day_avg_quantity', 'campaign3_day_avg_day_order_diff',
       'campaign3_day_avg_discount_ratio', 'campaign2_num_item_id',
       'campaign2_unique_item_id', 'campaign2_item_ratio',
       'campaign2_item_diff', 'campaign2_num_coupon_discount_applied',
       'campaign2_num_other_discount_applied', 'campaign2_ratio_num',
       'campaign2_brand', 'campaign2_mode_vc_brand', 'campaign2_brand_type',
       'campaign2_mode_vc_brand_type', 'campaign2_category',
       'campaign2_mode_vc_category', 'campaign2_unique_brand_id',
       'campaign2_count_brand_id', 'campaign2_brand_ratio',
       'campaign2_brand_diff', 'campaign2_avg_quantity',
       'campaign2_avg_selling_price',
       'campaign2_avg_other_discount_percentage',
       'campaign2_avg_coupon_discount_percentage', 'campaign2_day_avg_item_id',
       'campaign2_day_avg_other_discount_percentage',
       'campaign2_day_avg_selling_price',
       'campaign2_day_avg_coupon_discount_percentage',
       'campaign2_day_avg_quantity', 'campaign2_day_avg_day_order_diff',
       'campaign2_day_avg_discount_ratio']

random.drop(cols,axis=1,inplace=True)

ct_cmp_final=pd.merge(ct_cmp_type,random,on=['customer_id',
 'campaign_type'],how='left')
 

filename = "/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/int_data/campaign_type_feature"
outfile = open(filename,'wb')
pickle.dump(ct_cmp_final,outfile)
outfile.close()


pickle_in = open("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/int_data/new_version_feature","rb")
ct_cmp_id = pickle.load(pickle_in)


#combining all files and creating final features



import pickle as pkl

pickle_in = open("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/int_data/new_version_feature","rb")
ct_cmp_id = pickle.load(pickle_in)

pickle_in = open("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/int_data/campaign_type_feature","rb")
ct_cmp_type = pickle.load(pickle_in)

pickle_in = open("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/int_data/ct_feature_end_date","rb")
ct_cmp_id2 = pickle.load(pickle_in)

ct_cmp_final=pd.merge(ct_cmp_id,ct_cmp_id2,on=['customer_id','campaign_id'],how='left')

ct_cmp_type2=pd.merge(ct_cmp_type,campaign_data,on='campaign_type',how='left')

ct_final=pd.merge(ct_cmp_final,ct_cmp_type2,on=['customer_id','campaign_id'],how='left')

filename = "/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/int_data/ct_final"
outfile = open(filename,'wb')
pickle.dump(ct_final,outfile)
outfile.close()




pickle_in = open("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/int_data/attempt2","rb")
data_final1 = pickle.load(pickle_in)

pickle_in = open("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/int_data/ct_final","rb")
campaign_transaction = pickle.load(pickle_in)

data_final2=pd.merge(data_final1,campaign_transaction,on=['customer_id','campaign_id'],how='left')

data_divya=pkl.load(open('/axp/buanalytics/csgrbc/dev/Divya/AmExpert/data_nodup_features.pkl','rb'))

data_final2=pd.merge(data_final1,data_divya,on=[ 'customer_id','coupon_id','campaign_id'],how='left')

cols=['id','redemption_status',
       'campaign_type', 'start_date', 'end_date', 'item_id', 'brand',
       'brand_type', 'category', 'age_range', 'marital_status', 'rented',
       'family_size', 'no_of_children', 'income_bracket',
       'customer_id_confRate', 'campaign_type_confRate', 'coupon_id_confRate']
	   
data_divya.drop(cols,axis=1,inplace=True)

data_final3=pd.merge(data_final2,data_divya,on=[ 'customer_id','coupon_id','campaign_id'],how='left')

filename = "/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/int_data/ct_final_train_divya2"
outfile = open(filename,'wb')
pickle.dump(data_final3,outfile)
outfile.close()

cols=['customer_id','coupon_id']
for col in cols:
    data_final3['vc_'+col] = data_final3[col].map(data_final1[col].value_counts().to_dict())


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

cols=['campaign_type_x','campaign_type_y', 'age_range',
       'marital_status', 'rented', 'family_size', 'no_of_children',
       'income_bracket','brand', 'brand_type', 'category','mode_brand', 'mode_brand_type', 'mode_category']
for col in cols:
        le.fit(data_final3[col].values)
        data_final3[col]=le.transform(data_final3[col])


		
#model		
		

cols=[  'coupon_id', 'customer_id',
       'campaign_type', 
       'family_size',
       'income_bracket',
       'null_rented', 'null_family_size', 'distinct_item_id','distinct_category', 'count_item_id',
       'brand_ratio_x', 'brand_type_ratio', 'category_ratio',
       'mode_vc_brand_x', 'brand_type', 'mode_vc_brand_type_x', 'category',
       'mode_vc_category_x', 'campaign_length', 'unique_brand_id',
       'count_brand_id', 'brand_ratio_y', 'brand_diff',
       'unique_category', 'mode_vc_brand_y', 'mode_vc_brand_type_y',
       'mode_vc_category_y', 'num_coupon_discount_applied',
       'num_other_discount_applied', 'ratio_num', 'max_selling_price',
       'min_selling_price', 'num_item_id', 'unique_item_id', 'item_ratio',
       'item_diff', 'avg_quantity', 'avg_selling_price',
       'avg_other_discount_percentage', 'avg_coupon_discount_percentage','day_avg_item_id',
       'day_avg_other_discount_percentage', 'day_avg_selling_price',
       'day_avg_coupon_discount_percentage', 'day_avg_quantity',
       'day_avg_day_order_diff', 'diff_end_date_max', 'diff_end_date_min',
       'diff_end_date_mean', 'cmpg_cust_diff', 'cmpg_cust_ratio',
       'cmpg_cpn_diff', 'cmpg_cpn_ratio','cmpg_item_ratio',
       'cust_cmpg_diff', 'cust_cmpg_ratio', 'cust_cmpgt_diff',
       'cust_cmpgt_ratio', 'cust_cpn_diff', 'cust_cpn_ratio',
'cust_item_ratio', 'cpn_cmpgt_diff', 'cpn_cmpgt_ratio',
    'vc_customer_id', 'vc_coupon_id']

cols2=['coupon_id', 'customer_id',
       'campaign_type', 'age_range',
       'marital_status', 'rented', 'family_size', 'no_of_children',
       'income_bracket','brand', 'brand_type', 'category','mode_brand', 'mode_brand_type', 'mode_category']
	   
a=[train[cols].columns.get_loc(c) for c in cols2 if c in train[cols]]

categorical_features_indices=a

#catboost

train_cols=['coupon_id', 'customer_id','avg_discount_ratio','day_avg_coupon_discount_percentage',


'family_size'	,
'null_marital_status'	,


'distinct_item_id'	,

'count_item_id'	,
'count_brand'	,
'brand_ratio_x'	,
'brand_type_ratio'	,
'category_ratio'	,

'campaign_length'	,
'unique_brand_id'	,
'count_brand_id'	,
'brand_ratio_y'	,
'brand_diff'	,

'mode_vc_brand_y'	,
'mode_vc_brand_type_y'	,
'num_coupon_discount_applied'	,
'num_other_discount_applied'	,
'ratio_num'	,
'max_selling_price'	,
'min_selling_price'	,

'unique_item_id'	,
'item_ratio'	,
'item_diff'	,
'avg_quantity'	,
'avg_selling_price'	,

'avg_coupon_discount_percentage'	,

'day_avg_item_id'	,
'day_avg_other_discount_percentage'	,
'day_avg_selling_price'	,
'day_avg_coupon_discount_percentage'	,
'day_avg_quantity'	,
'day_avg_day_order_diff'	,
'diff_end_date_max'	,
'diff_end_date_min'	,
'diff_end_date_mean'	,
'campaign_num_item_id'	,

'campaign_item_ratio'	,

'campaign_num_coupon_discount_applied'	,
'campaign_num_other_discount_applied'	,
'campaign_ratio_num'	,


'campaign_brand_ratio'	,
'campaign_brand_diff'	,



'campaign_avg_coupon_discount_percentage'	,
'campaign_day_avg_item_id'	,

'campaign_day_avg_selling_price'	,
'campaign_day_avg_quantity'	,
'campaign_day_avg_day_order_diff'	,
'common_rationum_item_id'	,
'common_ratiounique_item_id'	,
'common_rationum_coupon_discount_applied'	,
'common_rationum_other_discount_applied'	,
'common_ratiounique_brand_id'	,
'common_ratioday_avg_item_id'	,
'common_ratioday_avg_quantity'	,
'common_ratioday_avg_day_order_diff'	,

'cmpg_cust_diff'	,
'cmpg_cust_ratio'	,
'cmpg_cpn_diff'	,
'cmpg_item_diff'	,
'cmpg_item_ratio'	,
'cust_cmpg_diff'	,
'cust_cmpg_ratio'	,

'cust_cmpgt_ratio'	,
'cust_cpn_diff'	,

'cust_item_diff'	,
'cust_item_ratio'	,
'cpn_cmpgt_diff'	,
'cpn_cmpgt_ratio'	,
'vc_customer_id'	,
'vc_coupon_id'	,
]

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import StratifiedKFold

def run_cv_model(train, test, target, model_fn,categorical_features_indices, params={}, eval_fn=None, label='model', n_folds=5):
    kf = StratifiedKFold(n_splits=n_folds, shuffle = True, random_state = 228)
    fold_splits = kf.split(train, target)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0]))
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = test.columns
    i = 1
    for dev_index, val_index in fold_splits:
        print('-------------------------------------------')
        #print('Started ' + label + ' fold ' + str(i) + {n_folds}).format(n_folds=n_folds)
        dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
        dev_y, val_y = target.iloc[dev_index], target.iloc[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y, fi = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        feature_importances['fold_{i}'.format(i=i)] = fi
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            print(label + ' cv score {}: {}'.format(i, cv_score), '\n')
        i += 1
    print('{} cv scores : {}'.format(label, cv_scores))
    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std score : {}'.format(label, np.std(cv_scores)))
    pred_full_test = pred_full_test / n_folds
    results = {'label': label,
              'train': pred_train, 'test': pred_full_test,
              'cv': cv_scores, 'fi': feature_importances}
    return results

def runCAT(train_X, train_y, test_X, test_y, test_X2, params):
    # Pool the data and specify the categorical feature indices
    print('Pool Data')
    _train = Pool(train_X, label=train_y,cat_features=categorical_features_indices)
    _valid = Pool(test_X, label=test_y,cat_features=categorical_features_indices)    
    print('Train CAT')
    model = CatBoostClassifier(**params)
    fit_model = model.fit(_train,
                          eval_set=_valid,
                          use_best_model=True,
                          verbose=1000,
                          plot=False)
    feature_im = fit_model.feature_importances_
    print('Predict 1/2')
    pred_test_y = fit_model.predict_proba(test_X)[:, 1]
    print('Predict 2/2')
    pred_test_y2 = fit_model.predict_proba(test_X2)[:, 1]
    return pred_test_y, pred_test_y2, feature_im


# Use some baseline parameters
cat_params = {'loss_function': 'CrossEntropy', 
              'eval_metric': "AUC",
              'learning_rate': 0.01,
              'iterations': 10000,
              'random_seed': 42,
              'od_type': "Iter",
              'early_stopping_rounds': 150,
             }

n_folds = 10

results = run_cv_model(train[cols].fillna(0), test[cols].fillna(0), train['redemption_status'], runCAT,categorical_features_indices, cat_params, auc, 'cat', n_folds=n_folds)

tmp = dict(zip(test.id_x.values, results['test']))
answer4 = pd.DataFrame()
answer4['id'] = test.id_x.values
answer4['redemption_status'] = answer4['id'].map(tmp)
answer4.shape
answer4.to_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/submission/submission_sun_9.csv",index=False)



#lightgbm

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_auc_score as auc
import lightgbm as lgb
from sklearn.metrics import roc_auc_score as auc
import seaborn as sns
import matplotlib.pyplot as plt


pickle_in = open("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/int_data/ct_final_train_divya_encoded2","rb")
data_final1 = pickle.load(pickle_in)

cols=[
'age_range'	,
'marital_status'	,
'rented'	,
'family_size'	,
'no_of_children'	,
'income_bracket'	,
'null_age_range'	,
'null_marital_status'	,
'null_rented'	,
'null_family_size'	,
'null_no_of_children'	,
'null_income_bracket'	,
'distinct_item_id'	,
'distinct_brand'	,
'distinct_brand_type'	,
'distinct_category'	,
'count_item_id'	,
'count_brand'	,
'count_brand_type'	,
'count_category'	,
'item_id_ratio'	,
'brand_ratio_x'	,
'brand_type_ratio'	,
'category_ratio'	,

'campaign_length'	,
'unique_brand_id'	,
'count_brand_id'	,
'brand_ratio_y'	,
'brand_diff'	,

'mode_vc_brand_y'	,
'mode_vc_brand_type_y'	,
'mode_vc_category_y'	,
'num_coupon_discount_applied'	,
'num_other_discount_applied'	,
'ratio_num'	,
'max_selling_price'	,
'min_selling_price'	,
'num_item_id'	,
'unique_item_id'	,
'item_ratio'	,
'item_diff'	,
'avg_quantity'	,
'avg_selling_price'	,
'avg_other_discount_percentage'	,
'avg_coupon_discount_percentage'	,

'day_avg_item_id'	,
'day_avg_other_discount_percentage'	,
'day_avg_selling_price'	,
'day_avg_coupon_discount_percentage'	,
'day_avg_quantity'	,
'day_avg_day_order_diff'	,
'diff_end_date_max'	,
'diff_end_date_min'	,
'diff_end_date_mean'	,
'campaign_num_item_id'	,
'campaign_unique_item_id'	,
'campaign_item_ratio'	,
'campaign_item_diff'	,
'campaign_num_coupon_discount_applied'	,
'campaign_num_other_discount_applied'	,
'campaign_ratio_num'	,

'campaign_mode_vc_brand'	,

'campaign_mode_vc_brand_type'	,

'campaign_mode_vc_category'	,
'campaign_unique_brand_id'	,
'campaign_count_brand_id'	,
'campaign_brand_ratio'	,
'campaign_brand_diff'	,
'campaign_avg_quantity'	,
'campaign_avg_selling_price'	,
'campaign_avg_other_discount_percentage'	,
'campaign_avg_coupon_discount_percentage'	,
'campaign_day_avg_item_id'	,
'campaign_day_avg_other_discount_percentage'	,
'campaign_day_avg_selling_price'	,
'campaign_day_avg_coupon_discount_percentage'	,
'campaign_day_avg_quantity'	,
'campaign_day_avg_day_order_diff'	,
'campaign_day_avg_discount_ratio'	,
'common_rationum_item_id'	,
'common_ratiounique_item_id'	,
'common_rationum_coupon_discount_applied'	,
'common_rationum_other_discount_applied'	,
'common_ratiounique_brand_id'	,
'common_ratiocount_brand_id'	,
'common_ratioday_avg_item_id'	,
'common_ratioday_avg_quantity'	,
'common_ratioday_avg_day_order_diff'	,

'cmpg_cust_diff'	,
'cmpg_cust_ratio'	,
'cmpg_cpn_diff'	,
'cmpg_cpn_ratio'	,
'cmpg_item_diff'	,
'cmpg_item_ratio'	,
'cust_cmpg_diff'	,
'cust_cmpg_ratio'	,
'cust_cmpgt_diff'	,
'cust_cmpgt_ratio'	,
'cust_cpn_diff'	,
'cust_cpn_ratio'	,
'cust_item_diff'	,
'cust_item_ratio'	,
'cpn_cmpgt_diff'	,
'cpn_cmpgt_ratio'	,
'vc_customer_id'	,
'vc_coupon_id'	,
]

train_cols=['coupon_id', 'customer_id','avg_discount_ratio','day_avg_coupon_discount_percentage',


'family_size'	,
'null_marital_status'	,


'distinct_item_id'	,

'count_item_id'	,
'count_brand'	,
'brand_ratio_x'	,
'brand_type_ratio'	,
'category_ratio'	,

'campaign_length'	,
'unique_brand_id'	,
'count_brand_id'	,
'brand_ratio_y'	,
'brand_diff'	,

'mode_vc_brand_y'	,
'mode_vc_brand_type_y'	,
'num_coupon_discount_applied'	,
'num_other_discount_applied'	,
'ratio_num'	,
'max_selling_price'	,
'min_selling_price'	,

'unique_item_id'	,
'item_ratio'	,
'item_diff'	,
'avg_quantity'	,
'avg_selling_price'	,

'avg_coupon_discount_percentage'	,

'day_avg_item_id'	,
'day_avg_other_discount_percentage'	,
'day_avg_selling_price'	,
'day_avg_coupon_discount_percentage'	,
'day_avg_quantity'	,
'day_avg_day_order_diff'	,
'diff_end_date_max'	,
'diff_end_date_min'	,
'diff_end_date_mean'	,
'campaign_num_item_id'	,

'campaign_item_ratio'	,

'campaign_num_coupon_discount_applied'	,
'campaign_num_other_discount_applied'	,
'campaign_ratio_num'	,


'campaign_brand_ratio'	,
'campaign_brand_diff'	,



'campaign_avg_coupon_discount_percentage'	,
'campaign_day_avg_item_id'	,

'campaign_day_avg_selling_price'	,
'campaign_day_avg_quantity'	,
'campaign_day_avg_day_order_diff'	,
'common_rationum_item_id'	,
'common_ratiounique_item_id'	,
'common_rationum_coupon_discount_applied'	,
'common_rationum_other_discount_applied'	,
'common_ratiounique_brand_id'	,
'common_ratioday_avg_item_id'	,
'common_ratioday_avg_quantity'	,
'common_ratioday_avg_day_order_diff'	,

'cmpg_cust_diff'	,
'cmpg_cust_ratio'	,
'cmpg_cpn_diff'	,
'cmpg_item_diff'	,
'cmpg_item_ratio'	,
'cust_cmpg_diff'	,
'cust_cmpg_ratio'	,

'cust_cmpgt_ratio'	,
'cust_cpn_diff'	,

'cust_item_diff'	,
'cust_item_ratio'	,
'cpn_cmpgt_diff'	,
'cpn_cmpgt_ratio'	,
'vc_customer_id'	,
'vc_coupon_id'	,
]



def get_importances(clfs):
    importances = [clf.feature_importance('gain') for clf in clfs]
    importances = np.vstack(importances)
    mean_gain = np.mean(importances, axis=0)
    features = clfs[0].feature_name()
    data = pd.DataFrame({'gain':mean_gain, 'feature':features})
    plt.figure(figsize=(8, 30))
    sns.barplot(x='gain', y='feature', data=data.sort_values('gain', ascending=False))
    plt.tight_layout()
    return data

def standart_split(data, n_splits):
    split_list = []
    for i in range(n_splits):
        kf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 228)
        for train_index, test_index in kf.split(data.iloc[:ltr, :], data['redemption_status'][:ltr]):
            split_list += [(train_index, test_index)]
    return split_list

split_list = standart_split(data_final1, 1)

def lgb_train(data, target, ltr, train_cols, split_list, param, n_e = 10000, cat_col = None, verb_num = None, imp=True):
    pred = pd.DataFrame()
    pred_val = np.zeros(ltr)
    score = []
    j = 0
    train_pred = pd.DataFrame()
    models = []
    for i , (train_index, test_index) in enumerate(split_list):
        param['seed'] = i
        tr = lgb.Dataset(np.array(data[train_cols])[train_index], np.array(data[target])[train_index])
        te = lgb.Dataset(np.array(data[train_cols])[test_index], np.array(data[target])[test_index], reference=tr)
        tt = lgb.Dataset(np.array(data[train_cols])[ltr:, :])
        evallist = [(tr, 'train'), (te, 'test')]
        bst = lgb.train(param, tr, num_boost_round = n_e,valid_sets = [tr, te], feature_name=train_cols,
                        early_stopping_rounds=150, verbose_eval = verb_num)
        pred[str(i)] =bst.predict(np.array(data[train_cols])[ltr:])
        pred_val[test_index] = bst.predict(np.array(data[train_cols])[test_index])
        score += [metrics.roc_auc_score(np.array(data[target])[test_index], pred_val[test_index])]
        models.append(bst)
        print(i, 'MEAN: ', np.mean(score), 'LAST: ', score[-1])
    if imp:
        get_importances(models)
        plt.show()
    train_pred[str(j)] = pred_val
    ans = pd.Series( pred.mean(axis = 1).tolist())
    ans.name = 'lgb'
    return pred, score, train_pred, bst

param_lgb = { 'boosting_type': 'gbdt', 'objective': 'binary', 'metric':'auc',
             'bagging_freq':1, 'subsample':1, 'feature_fraction': 0.7,
              'num_leaves': 8, 'learning_rate': 0.01, 'lambda_l1':5,'max_bin':255}


prediction, scores, oof, model = lgb_train(data_final1, 'redemption_status', ltr, train_cols,
                       split_list, param_lgb,  verb_num  = 250)


tmp = prediction.copy()
for col in tmp.columns:
    tmp[col] = tmp[col].rank()
tmp = tmp.mean(axis = 1)
tmp  =tmp / tmp.max()

tmp = dict(zip(test.id.values, tmp))
answer4= pd.DataFrame()
answer4['id'] = test.id.values
answer4['redemption_status'] = answer4['id'].map(tmp)
answer4.to_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/submission/submission_sun_6.csv",index=False)


#xgboost

#xgBOOST
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm_notebook as tqdm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


def xgb_train(data, target, ltr, train_cols, split_list, param, n_e = 10000, cat_col = None, verb_num = None, imp=False):
    pred = pd.DataFrame()
    pred_val = np.zeros(ltr)
    score = []
    j = 0
    train_pred = pd.DataFrame()
    models = []
    for i , (train_index, test_index) in enumerate(split_list):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        param['seed'] = i
        tr = xgb.DMatrix(sc.fit_transform(np.array(data[train_cols])[train_index]), np.array(data[target])[train_index])
        te = xgb.DMatrix(sc.fit_transform(np.array(data[train_cols])[test_index]), np.array(data[target])[test_index])
        tt = xgb.DMatrix(sc.fit_transform(np.array(data[train_cols]))[ltr:, :])
        evallist = [(tr, 'train'), (te, 'test')]
        bst = xgb.train(param, tr, n_e, evallist,
                        early_stopping_rounds=150, verbose_eval = verb_num)
        
        pred[str(i)] =bst.predict(tt)
        pred_val[test_index] = bst.predict(te)
        score += [metrics.roc_auc_score(np.array(data[target])[test_index], pred_val[test_index])]
        models.append(bst)
        print(i, 'MEAN: ', np.mean(score), 'LAST: ', score[-1])
    train_pred[str(j)] = pred_val
    ans = pd.Series( pred.mean(axis = 1).tolist())
    ans.name = 'xgb'
    return pred, score, train_pred, bst
params = {'eta': 0.05,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 0, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':100,
          'eval_metric': 'auc', 
          'nthread':8,
          'random_state': 99}

prediction, scores, oof, model = xgb_train(data_final1, 'redemption_status', ltr, train_cols,
                       split_list, params,  verb_num  = 250)


tmp = prediction.copy()
for col in tmp.columns:
    tmp[col] = tmp[col].rank()
tmp = tmp.mean(axis = 1)
tmp  =tmp / tmp.max()

tmp = dict(zip(test.id.values, tmp))
answer1 = pd.DataFrame()
answer1['id'] = test.id.values
answer1['redemption_status'] = answer1['id'].map(tmp)
answer1.to_csv("/axp/buanalytics/csgrbc/dev/Vishesh/analytics vidhya/AMexpert/submission/submission_sun_7.csv",index=False)







