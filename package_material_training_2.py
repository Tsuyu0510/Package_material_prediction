#!/usr/bin/env python
# coding: utf-8

# In[4]:


## get data 
#import numpy as np 
import pandas as pd 
import tensorflow as tf
import keras 
from tensorflow.keras.layers import Dense, Dropout
## label encoder 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df_raw_2 = pd.read_csv('package_material_train_2.csv')
df_raw_2 = df_raw_2.drop(['Unnamed: 0'],axis =1)
df_raw_2['系統推薦包材'] = labelencoder.fit_transform(df_raw_2['系統推薦包材'])
#df_raw_2 = pd.get_dummies(df_raw_2, columns=['系統推薦包材'])

df_raw_2 = pd.DataFrame(df_raw_2,columns=
                       ['訂單商品種類','體積','訂單商品訂購總數','最長邊_訂','次長邊_訂','訂單類型',
                        '破壞袋註記','系統推薦包材'])
df_raw_2


# In[5]:


train_data = df_raw_2.iloc[0:10001,:]
test_data = df_raw_2.iloc[10001:,:]
train_X = train_data.iloc[:,0:7]
train_Y = train_data.iloc[:,7:]
test_X = test_data.iloc[:,0:7]
test_Y = test_data.iloc[:,7:]
print (train_X)
print (train_Y)
print (test_X.shape)
print (test_Y.shape)


# In[6]:


import sklearn
from xgboost import XGBClassifier
xg1 = XGBClassifier(colsample_bytree= 0.3, learning_rate=0.01, max_depth= 5, n_estimators=1000)
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


# In[7]:


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




