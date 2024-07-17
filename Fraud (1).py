#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[2]:


def Processed_Chunk(chunk):
    chunk['amount'] = pd.to_numeric(chunk['amount'], errors ='coerce')
    chunk['oldbalanceOrg'] = pd.to_numeric(chunk['oldbalanceOrg'], errors ='coerce')
    chunk['newbalanceOrig'] = pd.to_numeric(chunk['newbalanceOrig'], errors ='coerce')
    chunk['oldbalanceDest'] = pd.to_numeric(chunk['oldbalanceDest'], errors ='coerce')
    chunk['newbalanceDest'] = pd.to_numeric(chunk['newbalanceDest'], errors ='coerce')
    chunk['isFraud'] = pd.to_numeric(chunk['isFraud'], errors ='coerce')
    chunk['nameOrig'] = pd.to_numeric(chunk['nameOrig'], errors ='coerce')
    chunk['nameDest'] = pd.to_numeric(chunk['nameDest'], errors ='coerce')



    chunk.fillna(0, inplace = True)
    LBE = LabelEncoder()
    chunk['type'] = LBE.fit_transform(chunk['type'])
    return chunk


# In[3]:


chunk_size = 100000
chunks = []


# In[4]:


for chunk in pd.read_csv("C:/Users/Ijaz khan/Downloads/Fraud.csv",chunksize=chunk_size):
    PC = Processed_Chunk(chunk)
chunks.append(PC)
df = pd.concat(chunks, axis = 0)


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# In[11]:


print(df.dtypes)


# In[12]:


df


# In[13]:


plt.figure(figsize= (12,10))
sns.boxplot(data=df)
plt.xticks(rotation = 90)
plt.show()


# In[14]:


from scipy import stats


# In[15]:


z_score = np.abs(stats.zscore(df[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']]))
threshold = 3

FO = (z_score < threshold).all(axis =1)
df = df[FO]
df


# In[16]:


print(df.dtypes)


# In[17]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score


# In[18]:


x = df[['step','type','amount','nameOrig','oldbalanceOrg','newbalanceOrig','nameDest','oldbalanceDest','newbalanceDest']]
y = df['isFraud']


# In[19]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)


# In[20]:


model = RandomForestClassifier(n_estimators = 100, random_state = 34)
model


# In[21]:


model.fit(x_train,y_train)


# In[22]:


y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)


# In[23]:


cm = confusion_matrix(y_test,y_pred)
print('\nconfusion_matrix :')
print(cm)


# In[35]:


cr = classification_report(y_test,y_pred)
print('\nclassification_report :')
print(cr)


# In[25]:


print('\nroc_auc_score :')
print(roc_auc_score(y_test,y_pred))


# In[26]:


data = ['step','type','amount','nameOrig','oldbalanceOrg','newbalanceOrig','nameDest','oldbalanceDest','newbalanceDest']


# In[27]:


feature_importances = pd.DataFrame(model.feature_importances_,index = data,
columns = ['importance']).sort_values('importance',ascending = False)
print(feature_importances)


# In[28]:


plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True, fmt = 'd', cmap = 'Blues',
xticklabels=['Not Fraud','Fraud'],
yticklabels=['Not Fraud','Fraud'])
plt.ylabel('Actual')
plt.xlabel('predicted')
plt.title(confusion_matrix)
plt.show()


# In[29]:


from sklearn.metrics import roc_curve, auc


# In[30]:


fpr, tpr, _ = roc_curve(y_test,y_pred)
roc_auc = auc(fpr,tpr)
plt.figure(figsize = (8,6))
plt.plot(fpr,tpr,color = 'darkorange',lw=2,
label = 'ROC curve (area = %0.2f)'% roc_auc)
plt.plot([0,1],[0,1],color = 'navy', lw=2,
linestyle = '--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('Fales Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[31]:


importances = model.feature_importances_
indices = np.argsort(importances)[::-1]


# In[32]:


plt.figure(figsize =(8,6))

plt.title("Feature Importance")
plt.bar(range(x_train.shape[1]), importances[indices],align = 'center')
plt.xticks(range(x_train.shape[1]),[data[i] for i in indices], rotation = 90)
plt.xlim([-1,x_train.shape[1]])
plt.show()


# In[ ]:





# In[ ]:




