#!/usr/bin/env python
# coding: utf-8

# In[1]:


## get data 
#import numpy as np 
from joblib import dump
import pickle
import pandas as pd 
import tensorflow as tf
import keras 
from tensorflow.keras.layers import Dense, Dropout
## label encoder 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()


        
        
df_raw_2 = pd.read_csv('package_material_train_2.csv')

        
        
df_raw_2

df_raw_2['紙箱註記'] = ''
for i in range(0,len(df_raw_2)):
    if df_raw_2['體積'][i] > 6300000 :
        df_raw_2['紙箱註記'][i] = 1
    else:
        df_raw_2['紙箱註記'][i] = 0

        


# In[ ]:





# In[2]:


df_raw_2 = df_raw_2.drop(['Unnamed: 0'],axis =1)
df_raw_2['系統推薦包材'] = labelencoder.fit_transform(df_raw_2['系統推薦包材'])
#df_raw_2 = pd.get_dummies(df_raw_2, columns=['系統推薦包材'])
df_raw_2['破壞袋利用率'] = df_raw_2['體積']/6300000
df_raw_2 = pd.DataFrame(df_raw_2,columns=
                       ['訂單商品種類','體積','破壞袋利用率','訂單商品訂購總數','最長邊_訂','次長邊_訂','訂單類型',
                        '破壞袋註記','紙箱註記','系統推薦包材'])
print(df_raw_2['系統推薦包材'].value_counts())
df_raw_2


# In[ ]:





# In[3]:


df_raw_2['紙箱註記'] = df_raw_2['紙箱註記'].astype(int)


# In[4]:


df_raw_3 = pd.read_csv('new_data_for_bags2boxs.csv')
df_raw_3 = df_raw_3.drop(['Unnamed: 0'],axis =1 )
## label encoder 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder() 
df_raw_3['實際過刷包裝'] = labelencoder.fit_transform(df_raw_3['實際過刷包裝'])
df_raw_3['實際過刷包裝'].value_counts()
df_raw_3 = df_raw_3.rename(columns = {'訂單商品種類_g':'訂單商品種類','體積_g':'體積','訂單商品訂購總數_g':'訂單商品訂購總數',
                           '最長邊_訂_g':'最長邊_訂','次長邊_訂_g':'次長邊_訂','訂單類型_g':'訂單類型','破壞袋註記_g':'破壞袋註記',
                            '實際過刷包裝':'系統推薦包材'})
df_raw_3


# In[18]:


df_raw_4 = pd.read_csv('202302_correct.csv')
df_raw_4 = df_raw_4.drop(['Unnamed: 0'],axis=1)
## label encoder 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder() 
df_raw_4['系統推薦包材'] = labelencoder.fit_transform(df_raw_4['系統推薦包材'])
print (df_raw_4['系統推薦包材'].value_counts())

df_raw_4 = df_raw_4.rename(columns = {'訂單商品種類_g':'訂單商品種類','體積_g':'體積','訂單商品訂購總數_g':'訂單商品訂購總數',
                           '最長邊_訂_g':'最長邊_訂','次長邊_訂_g':'次長邊_訂','訂單類型_g':'訂單類型','破壞袋註記_g':'破壞袋註記',
                            '實際過刷包裝':'系統推薦包材'})

df_raw_4


# In[19]:


df_raw_5 = pd.read_csv('202302_incorrect.csv')
df_raw_5 = df_raw_5.drop(['Unnamed: 0'],axis =1 )



df_raw_5 = df_raw_5.rename(columns = {'訂單商品種類_g':'訂單商品種類','體積_g':'體積','訂單商品訂購總數_g':'訂單商品訂購總數',
                           '最長邊_訂_g':'最長邊_訂','次長邊_訂_g':'次長邊_訂','訂單類型_g':'訂單類型','破壞袋註記_g':'破壞袋註記',
                            '實際過刷包裝':'系統推薦包材'})


## label encoder 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder() 
df_raw_5['系統推薦包材'] = labelencoder.fit_transform(df_raw_5['系統推薦包材'])
print (df_raw_5['系統推薦包材'].value_counts())


df_raw_5


# In[20]:


## concat 2 dfs 
df_concat_1_2 = pd.concat([df_raw_2,df_raw_3,df_raw_4,df_raw_5])
df_concat_1_2.reset_index(inplace=True)
df_concat_1_2 = df_concat_1_2.drop(['index'],axis =1)
df_concat_1_2


# In[21]:


from sklearn.model_selection import train_test_split

X = df_concat_1_2.iloc[:,0:9]
y = df_concat_1_2.iloc[:,9:]

train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.1, random_state=42)



#train_data = df_raw_2.iloc[0:10001,:]
#test_data = df_raw_2.iloc[10001:,:]
#train_X = train_data.loc[:,['訂單商品種類','體積','訂單商品訂購總數','最長邊_訂','次長邊_訂','訂單類型','破壞袋註記','紙箱註記']]

#train_Y = train_data.iloc[:,8:]
#test_X= test_data.loc[:,['訂單商品種類','體積','訂單商品訂購總數','最長邊_訂','次長邊_訂','訂單類型','破壞袋註記','紙箱註記']]
#test_Y = test_data.iloc[:,8:]
print (train_X)
print (train_Y)
print (test_X)
print (test_Y.shape)
print(train_X.dtypes)


# In[ ]:





# In[ ]:





# In[22]:


## try tree
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(train_X,train_Y)
tree_pred = clf_tree.predict(test_X)
from sklearn.metrics import confusion_matrix
cf_1 = confusion_matrix(test_Y,tree_pred)

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


print (sklearn.metrics.accuracy_score(test_Y, tree_pred)*100 )
print (sklearn.metrics.recall_score(test_Y, tree_pred,average='macro')*100)
print (sklearn.metrics.precision_score(test_Y, tree_pred,average='macro')*100)
print (sklearn.metrics.f1_score(test_Y, tree_pred,average='macro')*100)


# In[ ]:





# In[31]:


import sklearn
from xgboost import XGBClassifier
xg1 = XGBClassifier(colsample_bytree= 0.7, learning_rate=0.001, max_depth= 3, n_estimators=1000,objective='multi:softmax')
xg1=xg1.fit(train_X, train_Y)
predxgb = xg1.predict(test_X)
#print (predxgb)

from sklearn.metrics import confusion_matrix

cf_xgb = confusion_matrix(test_Y,predxgb)
cf_xgb

## confusion matrix
from sklearn.metrics import confusion_matrix
cf_1 = confusion_matrix(test_Y,predxgb)

## visuallize confusion matrix 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_xgb, display_labels = ['S53','S59','S78','S90','BAG'])

#ax = sns.heatmap(cf_xgb, annot=True, cmap='Blues')

#ax.set_title('Seaborn Confusion Matrix with labels');
#ax.set_xlabel('Predicted Values')
#ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
#ax.xaxis.set_ticklabels(['False','True'])
#ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.



#print (sklearn.metrics.recall_score(test_y, predxgb)*100)
#print (sklearn.metrics.precision_score(test_y, predxgb)*100)
#print (sklearn.metrics.f1_score(test_y, predxgb)*100)

cm_display.plot()

plt.show()
print (sklearn.metrics.accuracy_score(test_Y, predxgb)*100 )
print (sklearn.metrics.recall_score(test_Y, predxgb,average='macro')*100)
print (sklearn.metrics.precision_score(test_Y, predxgb,average='macro')*100)
print (sklearn.metrics.f1_score(test_Y, predxgb,average='macro')*100)

dump(xg1, 'xg1_bag.joblib')






# In[ ]:





# In[40]:


import xgboost as xgb

params = {
    "learning_rate": 0.001,
    "max_depth": 6,
    "num_class": 5,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softmax",
    "eval_metric": "auc",
    "colsample_bytree" :0.7,
    
}
dtrain = xgb.DMatrix(train_X, label=train_Y)
dvalid = xgb.DMatrix(test_X, label=test_Y)
model = xgb.train(params=params, dtrain=dtrain, num_boost_round=1000, 
                  early_stopping_rounds=50, evals=[(dtrain, "train"), (dvalid, "valid")], verbose_eval=10)



#dump(model, 'xgboost_formal.joblib')


# In[39]:


predxgb = model.predict(xgb.DMatrix(test_X))

#print (predxgb)

from sklearn.metrics import confusion_matrix

cf_xgb = confusion_matrix(test_Y,predxgb)
cf_xgb

## confusion matrix
from sklearn.metrics import confusion_matrix
cf_1 = confusion_matrix(test_Y,predxgb)

## visuallize confusion matrix 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_xgb, display_labels = ['S53','S59','S78','S90','BAG'])

#ax = sns.heatmap(cf_xgb, annot=True, cmap='Blues')

#ax.set_title('Seaborn Confusion Matrix with labels');
#ax.set_xlabel('Predicted Values')
#ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
#ax.xaxis.set_ticklabels(['False','True'])
#ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.



#print (sklearn.metrics.recall_score(test_y, predxgb)*100)
#print (sklearn.metrics.precision_score(test_y, predxgb)*100)
#print (sklearn.metrics.f1_score(test_y, predxgb)*100)

cm_display.plot()

plt.show()
print (sklearn.metrics.accuracy_score(test_Y, predxgb)*100 )
print (sklearn.metrics.recall_score(test_Y, predxgb,average='macro')*100)
print (sklearn.metrics.precision_score(test_Y, predxgb,average='macro')*100)
print (sklearn.metrics.f1_score(test_Y, predxgb,average='macro')*100)


# In[ ]:




