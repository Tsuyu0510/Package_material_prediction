#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
## get data 
raw_data = pd.read_excel('20230201_0228_correct.xlsx',engine="openpyxl")
raw_data


# In[18]:


#preprocess data 
## 加入包材尺寸

raw_data['體積'] = ''
raw_data['破壞袋註記'] =''
raw_data['系統推薦包材']= ''


## 加入包材成本

for i in range(0,len(raw_data)):
    
    raw_data['體積'][i]= raw_data['訂購數'][i]*raw_data['單品品項長'][i]*raw_data['單品品項寬'][i]*raw_data['單品品項高'][i]
    
    if raw_data['破壞袋'][i] ==1 :
        raw_data['破壞袋註記'][i] = 1
    else:
        raw_data['破壞袋註記'][i] = 0

    if raw_data['破壞袋'][i] == 1:
        raw_data['系統推薦包材'][i] = '破壞袋'
    elif raw_data['S53'][i] == 1:
        raw_data['系統推薦包材'][i] = 'S53'
    elif raw_data['S59'][i] == 1:
        raw_data['系統推薦包材'][i] = 'S59'
    elif raw_data['S78'][i] == 1:
        raw_data['系統推薦包材'][i] = 'S78'
    elif raw_data['S90'][i] == 1:
        raw_data['系統推薦包材'][i] = 'S90'
    else:
        raw_data['系統推薦包材'][i] = '未知'

        
raw_data = raw_data.drop(['破壞袋','S53','S59','S78','S90'],axis =1)
        
    
   



# In[20]:


#raw_data['破壞袋'].astype(int)
#raw_data['S53'].astype(int)
#raw_data['S59'].astype(int)
#raw_data['S78'].astype(int)
#raw_data['S90'].astype(int)
print (raw_data.dtypes)
raw_data


# In[21]:


raw_data.dtypes

raw_data_forjoin = pd.DataFrame(raw_data,columns=['廠商訂單編號','實際過刷包裝','系統推薦包材'])
raw_data_forjoin


# In[22]:


## 計算每件訂單中的商品訂購數量，作為判斷單品/複合單依據
df_group = raw_data.groupby(['廠商訂單編號'])['品名'].count()
## 計算每件訂單中的商品體積 
df_group_2 = raw_data.groupby(['廠商訂單編號'])['體積'].sum()
df_group_3 = raw_data.groupby(['廠商訂單編號'])['訂購數'].sum()
df_top = raw_data.groupby('廠商訂單編號')[['單品品項長', '單品品項寬', '單品品項高']].max()
#df_group_3 = raw_data.groupby(['廠商訂單編號'])['單品品項長','單品品項寬','單品品項高'].max()
#df_group_3 = raw_data.groupby(['廠商訂單編號'])[['單品品項長','單品品項寬','單品品項高']].apply(lambda x: x.nlargest(2, columns=['單品品項長','單品品項寬','單品品項高']))
df_group_4 =  raw_data.groupby(['廠商訂單編號'])['破壞袋註記'].sum()

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
df_group_MS_4 = pd.DataFrame(df_group_4)

#print (df_group_MS)
#print (df_group_MS_2)
df_group_MS_C_V = pd.concat([df_group_MS,df_group_MS_2,df_group_MS_3,df_group_MS_4],axis=1)
#df_group_MS_3
df_group_MS_C_V.reset_index(inplace=True)
df_group_MS_C_V = pd.concat([df_group_MS_C_V,df_package_size],axis =1 )
df_group_MS_C_V




# In[39]:


df_group_MS_C_V = df_group_MS_C_V.rename(columns= {'廠商訂單編號':'廠商訂單編號_g','品名':'訂單商品種類_g','訂購數':'訂單商品訂購總數_g','體積':'體積_g',
                                                   '最長邊_訂':'最長邊_訂_g','次長邊_訂':'次長邊_訂_g','破壞袋註記':'破壞袋註記_g'})
df_group_MS_C_V['訂單類型_g'] =''
df_group_MS_C_V['紙箱註記'] =''

df_group_MS_C_V


# In[44]:


for i in range(0,len(df_group_MS_C_V)):
    if df_group_MS_C_V['破壞袋註記_g'][i] < 1:
        df_group_MS_C_V['破壞袋註記_g'][i]  = 0 
    else:
        df_group_MS_C_V['破壞袋註記_g'][i] = 1
        
    if df_group_MS_C_V['訂單商品種類'][i] ==1 :
        df_group_MS_C_V['訂單類型_g'][i] = 0 ## 單品
    else:
        df_group_MS_C_V['訂單類型_g'][i] = 1 ## 複合單
    
    if df_group_MS_C_V['體積_g'][i] >6300000 :
        df_group_MS_C_V['紙箱註記'][i] = 1 ## 單品
    else:
        df_group_MS_C_V['紙箱註記'][i] = 0 ## 複合單
        
    


# In[ ]:





# In[48]:


## 合併訂單數量到原始數據
df_group_MS_C_V_join = df_group_MS_C_V.join(raw_data_forjoin.set_index('廠商訂單編號'),on='廠商訂單編號_g',how ='left')
df_group_MS_C_V_join
df_group_MS_C_V_join_d = df_group_MS_C_V_join.drop_duplicates(['廠商訂單編號_g','體積_g'])

df_group_MS_C_V_join_d['破壞袋利用率']= df_group_MS_C_V_join_d['體積_g']/6300000
df_group_MS_C_V_join_d = pd.DataFrame(df_group_MS_C_V_join_d,columns=['訂單商品種類','體積_g','破壞袋利用率','訂單商品訂購總數',
                                                                     '最長邊_訂_g','次長邊_訂_g','訂單類型_g','破壞袋註記_g',
                                                                     '紙箱註記','系統推薦包材']) 



# In[50]:


df_group_MS_C_V_join_d.to_csv('202302_correct.csv',encoding='utf-8')


# In[51]:


df_group_MS_C_V_join_d


# In[ ]:




