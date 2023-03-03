#!/usr/bin/env python
# coding: utf-8

# In[155]:


## get data 
#import numpy as np 
pd.set_option('display.max_columns',100) # 設定字元顯示寬度
pd.set_option('display.max_rows',100) # 設定字元顯示寬度

import pandas as pd 
import tensorflow as tf
import keras 
from tensorflow.keras.layers import Dense, Dropout


df_raw = pd.read_csv('package_material_train.csv')
print(df_raw.dtypes)


# In[156]:


import math

## 包材的推薦應不會隨時間有變化，主要應該還是跟訂單商品訂購有關。故後續建模會拿掉時間
df_raw_drop = df_raw.drop(['Unnamed: 0',
                           '包材使用率','全家商代','TS編號','最長邊','次長邊'],axis = 1 )

df_raw_drop = df_raw_drop.rename(columns={'品名_y': '訂單內商品種類','品名_x':'商品名稱'})
df_raw_drop_drop = df_raw_drop.drop(['破壞袋','S53','S59','S78','S90'],axis=1)
#df_raw_drop_drop['訂購日'] = pd.to_datetime(df_raw_drop_drop['訂購日']).astype(np.int64) ## convert timestamp to int
#df_raw_drop_drop['訂購日'] = math.log10(df_raw_drop_drop['訂購日'])


## label encoder 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df_raw_drop_drop['溫層'] = labelencoder.fit_transform(df_raw_drop_drop['溫層'])
df_raw_drop_drop['理貨DC'] = labelencoder.fit_transform(df_raw_drop_drop['理貨DC'])
df_raw_drop_drop['廠商訂單編號'] = labelencoder.fit_transform(df_raw_drop_drop['廠商訂單編號'])
df_raw_drop_drop['品名'] = labelencoder.fit_transform(df_raw_drop_drop['品名'])
#df_raw_drop_drop['TS編號'] = labelencoder.fit_transform(df_raw_drop_drop['TS編號'])
#df_raw_drop_drop['全家商代'] = labelencoder.fit_transform(df_raw_drop_drop['全家商代'])
df_raw_drop_drop['系統推薦包材'] = labelencoder.fit_transform(df_raw_drop_drop['系統推薦包材'])
#df_raw_drop_drop = pd.get_dummies(df_raw_drop_drop, columns=['系統推薦包材'])
import math
df_raw_drop_drop.columns
df_raw_drop_drop['包材體積'] = df_raw_drop_drop['包材長']*df_raw_drop_drop['包材寬']*df_raw_drop_drop['包材高']
df_raw_drop_drop = pd.DataFrame(df_raw_drop_drop,columns=['溫層', '理貨DC', '訂購日', '廠商訂單編號', '全台商代', '品名',
                                                          '訂購數', '單品品項長','單品品項寬', '單品品項高',
                                                          '破壞袋註記', '訂單商品種類', '體積_y', '訂單商品訂購總數', '單品/複合單',
                                                        '最長邊_品', '次長邊_品','系統推薦包材'])


#df_raw_drop_drop['體積_y'] = math.log10(df_raw_drop_drop['體積_y'])
#df_raw_drop_drop = df_raw_drop_drop.dropna()
print (df_raw_drop_drop.isnull().any())

df_raw_drop_drop


# In[157]:


#training_data = df_raw_drop_drop.iloc[0:18000,:]
#testing_data = df_raw_drop_drop.iloc[2000:,:]
training_data = df_raw_drop_drop[df_raw_drop_drop['訂購日'] < '2023-02-05']
testing_data = df_raw_drop_drop[df_raw_drop_drop['訂購日'] > '2023-02-04']
training_data = training_data.drop(['訂購日'],axis=1)
testing_data = testing_data.drop(['訂購日'],axis=1)


# In[158]:


train_x = training_data.iloc[:,0:16]
train_y = training_data.iloc[:,16:]
test_x = testing_data.iloc[:,0:16]
test_y = testing_data.iloc[:,16:]
print (train_x)
print (train_y)
print (test_x.shape)
print (test_y.shape)


# In[159]:


## try tree
from sklearn import tree
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(train_x,train_y)
tree_pred = clf_tree.predict(test_x)
from sklearn.metrics import confusion_matrix
cf_1 = confusion_matrix(test_y,tree_pred)

## visuallize confusion matrix 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import sklearn

get_ipython().run_line_magic('matplotlib', 'inline')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_1)

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


print (sklearn.metrics.accuracy_score(test_y, tree_pred)*100 )
print (sklearn.metrics.recall_score(test_y, tree_pred,average ='macro')*100)
print (sklearn.metrics.precision_score(test_y, tree_pred,average ='macro')*100)
print (sklearn.metrics.f1_score(test_y, tree_pred,average ='macro')*100)


# In[160]:


import sklearn
from xgboost import XGBClassifier
xg1 = XGBClassifier(colsample_bytree= 0.3, learning_rate=0.01, max_depth= 5, n_estimators=1000)
xg1=xg1.fit(train_x, train_y)
predxgb = xg1.predict(test_x)
#print (predxgb)

from sklearn.metrics import confusion_matrix

cf_xgb = confusion_matrix(test_y,predxgb)
cf_xgb

## confusion matrix
from sklearn.metrics import confusion_matrix
cf_1 = confusion_matrix(test_y,predxgb)

## visuallize confusion matrix 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_xgb, display_labels = [0,1,2,3,4])

#ax = sns.heatmap(cf_xgb, annot=True, cmap='Blues')

#ax.set_title('Seaborn Confusion Matrix with labels');
#ax.set_xlabel('Predicted Values')
#ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
#ax.xaxis.set_ticklabels(['False','True'])
#ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.



print (sklearn.metrics.accuracy_score(test_y, predxgb)*100 )
print (sklearn.metrics.recall_score(test_y, predxgb,average ='micro')*100)
print (sklearn.metrics.precision_score(test_y, predxgb,average ='micro')*100)
print (sklearn.metrics.f1_score(test_y, predxgb,average ='micro')*100)

cm_display.plot()

plt.show()


# In[161]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=256)
classifier.fit(train_x, train_y)
knn_pred = classifier.predict(test_x)
cf  = confusion_matrix(test_y,knn_pred)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf, display_labels = [0,1,2,3,4])
cm_display.plot()

plt.show()


print (sklearn.metrics.accuracy_score(test_y, knn_pred)*100 )
print (sklearn.metrics.recall_score(test_y, knn_pred,average ='micro')*100)
print (sklearn.metrics.precision_score(test_y, knn_pred,average ='micro')*100)
print (sklearn.metrics.f1_score(test_y, knn_pred,average ='micro')*100)


# In[162]:


from sklearn import svm
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(train_x, train_y)
svm_pred = clf.predict(test_x)
cf_svm = confusion_matrix(test_y,svm_pred)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_svm, display_labels = [0,1,2,3,4])
cm_display.plot()

plt.show()


print (sklearn.metrics.accuracy_score(test_y, svm_pred)*100 )
print (sklearn.metrics.recall_score(test_y, svm_pred,average ='micro')*100)
print (sklearn.metrics.precision_score(test_y, svm_pred,average ='micro')*100)
print (sklearn.metrics.f1_score(test_y, svm_pred,average ='micro')*100)


# In[164]:


'''from keras.models import Sequential
from keras.layers import Dense

# 創建一個序列模型
model = Sequential()

# 添加一個輸入層和兩個隱藏層
model.add(Dense(units=200, activation='relu', input_dim = train_x.shape[1]))
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=32, activation='relu'))


# 添加一個輸出層，使用 softmax 激活函數進行多分類預測
#model.add(Dense(units=10, activation='softmax'))
model.add(Dense(train_y.shape[1] , activation = 'softmax'))


# 編譯模型，指定損失函數、優化器和評估指標
model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.summary()


epochs = 20 
# 訓練模型
history = model.fit(train_x, train_y, epochs= epochs, batch_size=8,validation_data=(test_x,test_y))

# 用測試集評估模型
score = model.evaluate(test_x, test_y, batch_size=8)
print (score)

# 進行預測
#y_pred = model.predict(test_x)
#print (y_pred)
'''


# In[ ]:


import matplotlib.pyplot as plt
accuracy = history.history['accuracy']
loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']
epochs = range(epochs)
plt.plot(epochs,accuracy,'r-')
plt.plot(epochs,val_accuracy,'b-')
plt.legend()
plt.show()

plt.plot(epochs,loss,'r-')
plt.plot(epochs,val_loss,'b-')
plt.legend()
plt.show()


# In[ ]:


df_pred = pd.DataFrame(y_pred,columns=test_y.columns)
df_pred


# In[40]:


def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df
final_result_add_1 = undummify(df_pred)
#final_result_add.sort_values(by='concate')
#final_result_add.to_csv('20220719_test.csv')
final_result_add_1 ## predict result 


# In[ ]:


final_result_add_2 = undummify(test_y)
final_result_add_2.reset_index(inplace=True)
final_result_add_2 = final_result_add_2.drop(['index'],axis =1 )
final_result_add_2 = final_result_add_2.rename(columns={'系統推薦包材': '系統推薦包材_ground_truth'})
final_result_add_2


# In[ ]:


df_concat = pd.concat([final_result_add_1,final_result_add_2],axis =1) 
df_concat


# In[138]:


correct_count = 0
wrong_count = 0
i=0
for i in range(len(df_concat)):
    if df_concat['系統推薦包材'][i] == df_concat['系統推薦包材_ground_truth'][i]:
        correct_count +=1
    else:
        wrong_count+=1

        
print(correct_count)
print(wrong_count)
print (correct_count/len(df_concat))        


# 

# In[ ]:





# In[154]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




