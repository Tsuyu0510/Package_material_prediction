#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 

## get data 
raw_data = pd.read_excel('package_validationdata.xlsx',engine="openpyxl")
#raw_data = raw_data.dropna(subset='實際過刷包裝')


# In[2]:


raw_data['實際過刷包裝'].isnull().any()


# In[3]:


#preprocess data 
## 加入包材尺寸

raw_data['體積'] = ''
raw_data['破壞袋註記'] =''
raw_data['包材使用率'] = ''
raw_data['紙箱註記'] = ''


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
        
    if raw_data['實際過刷包裝'][i] == '大袋':
        raw_data['實際過刷包裝'][i] = '袋'

    if raw_data['體積'][i] > 6300000:
        raw_data['紙箱註記'][i] = 1 ## 必須用紙箱，因為破壞袋裝不下
    else:
        raw_data['紙箱註記'][i] = 0 ## 可以用破壞袋
        
    
   



# In[4]:



pd.set_option('display.max_rows',100) # 設定字元顯示寬度

raw_data['系統推薦包材']= ''

for i in range(0,len(raw_data)):
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
raw_data


# In[5]:


raw_data.dtypes

raw_data_forjoin = pd.DataFrame(raw_data,columns=['廠商訂單編號','實際過刷包裝','紙箱註記','系統推薦包材'])
raw_data_forjoin


# In[6]:


## 計算每件訂單中的商品訂購數量，作為判斷單品/複合單依據
df_group = raw_data.groupby(['廠商訂單編號'])['品名'].count()
## 計算每件訂單中的商品體積 
df_group_2 = raw_data.groupby(['廠商訂單編號'])['體積'].sum()
df_group_3 = raw_data.groupby(['廠商訂單編號'])['訂購數'].sum()
df_top = raw_data.groupby('廠商訂單編號')[['單品品項長', '單品品項寬', '單品品項高']].max()
df_group_4 =  raw_data.groupby(['廠商訂單編號'])['破壞袋註記'].sum()
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
    
    
#for j in range(0,len(raw_data)):
#    if df_group_4[j]<1:
#        raw_data['破壞袋註記_2'][j] = '破壞袋'
#    else:
#        raw_data['破壞袋註記_2'][j] = '紙箱'

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



# In[7]:


df_group_MS_C_V = df_group_MS_C_V.rename(columns= {'廠商訂單編號':'廠商訂單編號_g','品名':'訂單商品種類_g','訂購數':'訂單商品訂購總數_g','體積':'體積_g',
                                                   '最長邊_訂':'最長邊_訂_g','次長邊_訂':'次長邊_訂_g','破壞袋註記':'破壞袋註記_g'})
df_group_MS_C_V['訂單類型_g'] =''
df_group_MS_C_V


# In[8]:


for i in range(0,len(df_group_MS_C_V)):
    if df_group_MS_C_V['破壞袋註記_g'][i] < 1:
        df_group_MS_C_V['破壞袋註記_g'][i]  = 0 
    else:
        df_group_MS_C_V['破壞袋註記_g'][i] = 1
    
    
    
    if df_group_MS_C_V['訂單商品種類_g'][i] > 1:
        df_group_MS_C_V['訂單類型_g'][i] = 1 ## 復合單
    else:
        df_group_MS_C_V['訂單類型_g'][i] = 0 ## 單品單

        
df_group_MS_C_V['破壞袋註記_g'].value_counts()
df_group_MS_C_V['訂單類型_g'].value_counts()


# In[9]:


## get 系統推薦和實際過刷結果 join goruopby 結果

df_group_MS_C_V_join = df_group_MS_C_V.join(raw_data_forjoin.set_index('廠商訂單編號'),on='廠商訂單編號_g',how ='left')
df_group_MS_C_V_join
df_group_MS_C_V_join_d = df_group_MS_C_V_join.drop_duplicates(['廠商訂單編號_g','體積_g'])
df_group_MS_C_V_join_d


# In[ ]:





        


# In[10]:


print(df_group_MS_C_V_join_d['訂單商品種類_g'].value_counts()) 
print(df_group_MS_C_V_join_d['體積_g'].value_counts())
print(df_group_MS_C_V_join_d['訂單類型_g'].value_counts())


# In[11]:


df_group_MS_C_V_join_d['體積_g'] = df_group_MS_C_V_join_d['體積_g'].astype(int)
#df_group_MS_C_V_join_d['訂單類型_g'] = df_group_MS_C_V_join_d['訂單類型_g'].astype(int)
df_group_MS_C_V_join_d['破壞袋註記_g'] = df_group_MS_C_V_join_d['破壞袋註記_g'].astype(int)
df_group_MS_C_V_join_d['紙箱註記'] = df_group_MS_C_V_join_d['紙箱註記'].astype(int)
df_group_MS_C_V_join_d['訂單類型_g'] = df_group_MS_C_V_join_d['訂單類型_g'].astype(int)
df_group_MS_C_V_join_d['破壞袋利用率'] = df_group_MS_C_V_join_d['體積_g']/6300000
print(df_group_MS_C_V_join_d.dtypes)
print (df_group_MS_C_V_join_d.shape)
df_group_MS_C_V_join_d = pd.DataFrame(df_group_MS_C_V_join_d,columns=['廠商訂單編號_g','訂單商品種類_g','體積_g','破壞袋利用率','訂單商品訂購總數_g',
                                                                     '最長邊_訂_g','次長邊_訂_g','訂單類型_g','破壞袋註記_g','紙箱註記',
                                                                     '實際過刷包裝','系統推薦包材'])

## label encoder 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()        
df_group_MS_C_V_join_d['系統推薦包材'] = labelencoder.fit_transform(df_group_MS_C_V_join_d['系統推薦包材'])
#df_group_MS_C_V_join_d['實際過刷包裝'] = labelencoder.fit_transform(df_group_MS_C_V_join_d['實際過刷包裝'])


    
df_group_MS_C_V_join_d = df_group_MS_C_V_join_d.dropna(subset='實際過刷包裝')
df_group_MS_C_V_join_d['實際過刷包裝'] = labelencoder.fit_transform(df_group_MS_C_V_join_d['實際過刷包裝'])

df_group_MS_C_V_join_d
        


# In[24]:


df_group_MS_C_V_join_d 




# In[25]:


df_group_MS_C_V_join_d = df_group_MS_C_V_join_d.rename(columns={'廠商訂單編號_g':'廠商訂單編號','訂單商品種類_g':'訂單商品種類','體積_g':'體積','破壞袋利用率':'破壞袋利用率',
                                                                '訂單商品訂購總數_g':'訂單商品訂購總數','最長邊_訂_g':'最長邊_訂','次長邊_訂_g':'次長邊_訂','訂單類型_g':'訂單類型',
                                                                '破壞袋註記_g':'破壞袋註記','紙箱註記':'紙箱註記'})


# In[26]:


val_x = df_group_MS_C_V_join_d.loc[:,['訂單商品種類','體積','破壞袋利用率','訂單商品訂購總數','最長邊_訂','次長邊_訂','訂單類型','破壞袋註記','紙箱註記']]
val_y = df_group_MS_C_V_join_d.iloc[:,11:]
val_y_2 = df_group_MS_C_V_join_d['實際過刷包裝']
#new_data = np.array([[val_x]])
#new_data
print (val_x)
print (val_y.value_counts())
print (val_y_2.value_counts())


# In[27]:


from joblib import load
import pickle

clf_bag = load('xg1_bag.joblib')
print (clf_bag)
y_pred = clf_bag.predict(val_x)



# In[28]:


xgb_pred = pd.DataFrame(y_pred)
xgb_pred.value_counts()
## 0-> bag, 1->S53 , 2->S59, 3->S78, 4->S90


# In[29]:


## 系統推薦 VS 模型預估

from sklearn.metrics import confusion_matrix
cf_1 = confusion_matrix(val_y,y_pred)

## visuallize confusion matrix 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn

get_ipython().run_line_magic('matplotlib', 'inline')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_1,display_labels = ['S53','S59','S78','S90','BAG'])

#ax = sns.heatmap(cf_xgb, annot=True, cmap='Blues')

#ax.set_title('Seaborn Confusion Matrix with labels');
#ax.set_xlabel('Predicted Values')
#ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
#ax.xaxis.set_ticklabels(['False','True'])
#ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
cm_display.plot()

plt.show()


print (sklearn.metrics.accuracy_score(val_y, xgb_pred)*100 )
print (sklearn.metrics.recall_score(val_y, xgb_pred,average='macro')*100)
print (sklearn.metrics.precision_score(val_y, xgb_pred,average='macro')*100)
print (sklearn.metrics.f1_score(val_y, xgb_pred,average='macro')*100)


# In[30]:


## 實際過刷 vs 模型預估

from sklearn.metrics import confusion_matrix
cf_1 = confusion_matrix(val_y_2,y_pred)

## visuallize confusion matrix 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn

get_ipython().run_line_magic('matplotlib', 'inline')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_1,display_labels = ['S53','S59','S78','S90','BAG'])

#ax = sns.heatmap(cf_xgb, annot=True, cmap='Blues')

#ax.set_title('Seaborn Confusion Matrix with labels');
#ax.set_xlabel('Predicted Values')
#ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
#ax.xaxis.set_ticklabels(['False','True'])
#ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
cm_display.plot()

plt.show()


print (sklearn.metrics.accuracy_score(val_y_2, xgb_pred)*100 )
print (sklearn.metrics.recall_score(val_y_2, xgb_pred,average='macro')*100)
print (sklearn.metrics.precision_score(val_y_2, xgb_pred,average='macro')*100)
print (sklearn.metrics.f1_score(val_y_2, xgb_pred,average='macro')*100)


# In[32]:


## xgb.DMatrix data fromat 

from joblib import load
import pickle
import xgboost as xgb

#predxgb = model.predict(xgb.DMatrix(test_X))

clf_bag = load('xgboost_formal.joblib')

## need to transfer data to DMatrix object


print (clf_bag)
y_pred = clf_bag.predict(xgb.DMatrix(val_x))

cf_1 = confusion_matrix(val_y,y_pred)

## visuallize confusion matrix 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn

get_ipython().run_line_magic('matplotlib', 'inline')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_1,display_labels = ['S53','S59','S78','S90','BAG'])

#ax = sns.heatmap(cf_xgb, annot=True, cmap='Blues')

#ax.set_title('Seaborn Confusion Matrix with labels');
#ax.set_xlabel('Predicted Values')
#ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
#ax.xaxis.set_ticklabels(['False','True'])
#ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
cm_display.plot()

plt.show()


print (sklearn.metrics.accuracy_score(val_y, xgb_pred)*100 )
print (sklearn.metrics.recall_score(val_y, xgb_pred,average='macro')*100)
print (sklearn.metrics.precision_score(val_y, xgb_pred,average='macro')*100)
print (sklearn.metrics.f1_score(val_y, xgb_pred,average='macro')*100)


# In[ ]:





# In[ ]:





# In[ ]:




