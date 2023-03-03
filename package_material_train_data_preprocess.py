#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np 
import pandas as pd 


# In[31]:


## get data 
raw_data = pd.read_excel('9weeks.xlsx',engine="openpyxl")
raw_data


# In[32]:


#preprocess data 
## 加入包材尺寸

raw_data['體積'] = ''
raw_data['最長邊']= ''
raw_data['次長邊'] = ''
raw_data['破壞袋註記'] =''
raw_data['包材長'] = ''
raw_data['包材寬'] = ''
raw_data['包材高'] = ''
raw_data['包材體積'] = ''
raw_data['包材使用率'] = ''

## 加入包材成本

for i in range(0,len(raw_data)):
    
    if raw_data['破壞袋'][i] == 1.1:
        raw_data['破壞袋'][i] = 1
        
    if raw_data.iloc[i,11] == 0.1:
        raw_data['S59'][i] = 0
    
    if raw_data['S78'][i] == 0.2:
        raw_data['S78'][i] = 0
    
    
    if raw_data['S90'][i] == 0:
        raw_data['S90'][i] = 0
    
    elif raw_data['S90'][i] == 1:
        raw_data['S90'][i] = 1
    
    elif raw_data['S90'][i] == 0.3:
        raw_data['S90'][i] = 0
    else: 
        raw_data['S90'][i] = 1
    
    raw_data['體積'][i]= raw_data['訂購數'][i]*raw_data['單品品項長'][i]*raw_data['單品品項寬'][i]*raw_data['單品品項高'][i]
    
    if raw_data['破壞袋'][i] ==1 :
        raw_data['破壞袋註記'][i] = 1
    else:
        raw_data['破壞袋註記'][i] = 0
        
    
   



# In[33]:


raw_data['破壞袋'].astype(int)
raw_data['S53'].astype(int)
raw_data['S59'].astype(int)
raw_data['S78'].astype(int)
raw_data['S90'].astype(int)
print (raw_data.dtypes)
raw_data


# In[64]:


## 計算每件訂單中的商品訂購數量，作為判斷單品/複合單依據
df_group = raw_data.groupby(['廠商訂單編號'])['品名'].count()
## 計算每件訂單中的商品體積 
df_group_2 = raw_data.groupby(['廠商訂單編號'])['體積'].sum()
df_group_3 = raw_data.groupby(['廠商訂單編號'])['訂購數'].sum()
df_top = raw_data.groupby('廠商訂單編號')[['單品品項長', '單品品項寬', '單品品項高']].max()
#df_group_3 = raw_data.groupby(['廠商訂單編號'])['單品品項長','單品品項寬','單品品項高'].max()
#df_group_3 = raw_data.groupby(['廠商訂單編號'])[['單品品項長','單品品項寬','單品品項高']].apply(lambda x: x.nlargest(2, columns=['單品品項長','單品品項寬','單品品項高']))
raw_list = df_top.values.tolist()
max_list = []
second_list = []
i=0
for i in range(0,len(raw_list)):
    raw_list[i].sort()
    max_list.append(raw_list[i][-1])
    second_list.append(raw_list[i][-2])
    
    i+=1

df_max_list = pd.DataFrame(max_list,columns= ['最長邊_訂'])
df_second_list = pd.DataFrame(second_list,columns= ['次長邊_訂'])
df_package_size = pd.concat([df_max_list,df_second_list],axis = 1)


df_group_MS = pd.DataFrame(df_group)
df_group_MS_2 = pd.DataFrame(df_group_2)
df_group_MS_3 = pd.DataFrame(df_group_3)
#print (df_group_MS)
#print (df_group_MS_2)
df_group_MS_C_V = pd.concat([df_group_MS,df_group_MS_2,df_group_MS_3],axis=1)
#df_group_MS_3
df_group_MS_C_V.reset_index(inplace=True)
df_group_MS_C_V = pd.concat([df_group_MS_C_V,df_package_size],axis =1 )
df_group_MS_C_V




# In[65]:


df_group_MS_C_V = df_group_MS_C_V.rename(columns= {'廠商訂單編號':'廠商訂單編號','品名':'訂單商品種類','訂購數':'訂單商品訂購總數'})
df_group_MS_C_V


# In[66]:


## 合併訂單數量到原始數據
raw_data_join = pd.merge(raw_data,df_group_MS_C_V, on='廠商訂單編號',how = 'left')
raw_data_join ['單品/複合單'] =''

raw_data_join
for i in range(0,len(raw_data_join)):
    if raw_data_join['訂單商品種類'][i] ==1 :
        raw_data_join['單品/複合單'][i] = 0
    else:
        raw_data_join['單品/複合單'][i] = 1
    
    


# In[67]:


raw_data_join.dtypes


# In[68]:


raw_data_join



# In[ ]:





# In[39]:


raw_data_join_drop = raw_data_join.drop(['箱號','體積_x','QC人員','包材長','包材寬','包材高','包材體積'],axis =1)
raw_data_join_drop
#normalize 體積
from sklearn.preprocessing import StandardScaler
import math
# 創建 StandardScaler 物件
#scaler = StandardScaler()
#raw_data_join_drop['體積_y'] = scaler.fit_transform(raw_data_join_drop[['體積_y']])
#raw_data_join_drop['單品品項長'] = scaler.fit_transform(raw_data_join_drop[['單品品項長']])
#raw_data_join_drop['單品品項寬'] = scaler.fit_transform(raw_data_join_drop[['單品品項寬']])
#raw_data_join_drop['單品品項高'] = scaler.fit_transform(raw_data_join_drop[['單品品項高']])
#raw_data_join_drop['體積_y'] = math.log10(raw_data_join_drop['體積_y'])
#raw_data_join_drop['單品品項長'] = math.log10(raw_data_join_drop['單品品項長'])
#raw_data_join_drop['單品品項寬'] = math.log10(raw_data_join_drop['單品品項寬'])
#raw_data_join_drop['單品品項高'] = math.log10(raw_data_join_drop['單品品項高'])
#raw_data_join_drop['體積_y'] = math.log10(raw_data_join_drop['體積_y'])


# 標準化數據
raw_data_join_drop


# In[26]:


pd.set_option('display.max_rows',100) # 設定字元顯示寬度

raw_data_join_drop['系統推薦包材']= ''

for i in range(0,len(raw_data_join_drop)):
    if raw_data_join_drop['破壞袋'][i] == 1:
        raw_data_join_drop['系統推薦包材'][i] = '破壞袋'
    elif raw_data_join_drop['S53'][i] == 1:
        raw_data_join_drop['系統推薦包材'][i] = 'S53'
    elif raw_data_join_drop['S59'][i] == 1:
        raw_data_join_drop['系統推薦包材'][i] = 'S59'
    elif raw_data_join_drop['S78'][i] == 1:
        raw_data_join_drop['系統推薦包材'][i] = 'S78'
    elif raw_data_join_drop['S90'][i] == 1:
        raw_data_join_drop['系統推薦包材'][i] = 'S90'
    else:
        raw_data_join_drop['系統推薦包材'][i] = '未知'
raw_data_join_drop


    


# In[27]:


## 包材資訊df

data = {'實際過刷包裝': ['大袋','袋', 'S53', 'S59', 'S78','S90'],
        '包材長': [400, 335, 280, 280,380,370],
        '包材寬': [450, 350, 190, 190,280,270],
        '包材高': [1,1,75,120,120,240]}
df_material = pd.DataFrame(data)
df_material


# In[28]:


## join to raw data 
raw_data_join_drop_d = pd.merge(raw_data_join_drop,df_material, on='實際過刷包裝',how = 'left')
raw_data_join_drop_d


# In[29]:


## make df for commodity 
df_commodity = pd.DataFrame(raw_data_join_drop_d,columns=['單品品項長','單品品項寬','單品品項高'])
raw_list = df_commodity.values.tolist()
max_list = []
second_list = []
i=0
for i in range(0,len(raw_list)):
    raw_list[i].sort()
    max_list.append(raw_list[i][-1])
    second_list.append(raw_list[i][-2])
    
    i+=1

df_max_list = pd.DataFrame(max_list,columns= ['最長邊_品'])
df_second_list = pd.DataFrame(second_list,columns= ['次長邊_品'])
df_package_size = pd.concat([df_max_list,df_second_list],axis = 1)

raw_data_join_drop_d_d = pd.concat([raw_data_join_drop_d,df_package_size],axis=1)

raw_data_join_drop_d_d


# In[15]:


raw_data_join_drop_d_d.to_csv('package_material_train.csv')


# ### another approach 一行row 是一筆記錄，只考慮訂單編號和總體積，商品總數

# In[69]:


df_group_MS_C_V
df_group_MS_C_V['訂單類型'] = ''

for i in range(0,len(df_group_MS_C_V)):
    if df_group_MS_C_V['訂單商品種類'][i] == 1:
        df_group_MS_C_V['訂單類型'][i]  = 0 ## 單品單
    else:
        df_group_MS_C_V['訂單類型'][i] = 1 ## 復合單
        


# In[70]:


df_group_MS_C_V


# In[ ]:





# In[72]:


pd.set_option('display.max_rows',100) # 設定字元顯示寬度

df_test_big = pd.DataFrame(raw_data_join_drop_d_d,columns= ['廠商訂單編號','系統推薦包材','破壞袋註記','體積_y'])
df_test_big['體積'] =  df_test_big['體積_y'] 
df_test_big = df_test_big.drop(['體積_y'],axis=1)
df_test_big


# In[78]:


test_join_df = pd.merge(df_group_MS_C_V,df_test_big, on=['廠商訂單編號','體積'],how = 'left')
test_join_df = test_join_df.drop_duplicates()


        

test_join_df


# In[79]:


test_join_df.to_csv('package_material_train_2.csv')


# In[ ]:





# In[ ]:




