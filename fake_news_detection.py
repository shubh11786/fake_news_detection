#!/usr/bin/env python
# coding: utf-8

# ## IMPORTING REQUIRED LIBRARIES

# In[152]:


import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# ## DATA LOADING

# In[252]:


# Import dataset
df=pd.read_csv('C:\\Users\\shubham verma\\Downloads\\train.csv')

# Get the shape
df.shape


# In[253]:


df.head(10)


# ## DATA CLEANSING

# ### REMOVING NULL

# In[254]:


df.isnull().sum()


# In[255]:


df = df[['text','label']]


# In[256]:


df = df.dropna()


# In[257]:


df.isnull().sum()


# In[258]:


df.shape


# In[259]:


df['text'] = df['text'].apply(lambda x: x.lower())


# ### REMOVE PUNCTUATIONS

# In[260]:


import string
string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~‘’”“—' # manually adding unwanted punctuations
def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_txt = ''.join(all_list)
    return clean_txt
df['text'] = df['text'].apply(punctuation_removal)


# ### REMOVE STOPWORDS

# In[261]:


import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
def stopwords_removal(text):
# NEW LIST TO REPLACE STOPWORDS MANUALLY HAVING ' ’ ' i.e(where don't is present as don’t)
    stop_words = ["aren't",'couldn',"couldn't","don't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"should've","shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",'wouldn',"wouldn't"]
    stop = stopwords.words('english')
    stop_words=[x.replace("'","’") for x in stop_words]
    [stop.append(x) for x in stop_words]
    clean_txt = ' '.join([word for word in text.split() if word not in (stop)])
    return clean_txt
df['text'] = df['text'].apply(stopwords_removal)


# ### LEMMATIZATION

# In[32]:


import spacy


# In[33]:


sp = spacy.load('en_core_web_sm')


# In[262]:


# LEMMATIZATION OF EACH WORD IN NEWS ARTICLE.
df['text'] = df['text'].apply(lambda x: ' '.join([word.lemma_ for word in sp(x)]))


# In[110]:


# df['text'][0]


# ## TRAIN TEST SPLIT

# In[263]:


# splitting the data into train and test sets
x_train,x_test,y_train,y_test=train_test_split(df['text'], df.label, test_size=0.2, random_state=123)


# ## TF-IDF  

# In[264]:


#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)


# In[265]:


# Fit & transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# ## LOGISTIC REGRESSION

# In[266]:


from sklearn.linear_model import LogisticRegression


# In[267]:


clf = LogisticRegression(random_state=42,C=1)
clf.fit(tfidf_train,y_train)


# In[268]:


# Predict and calculate accuracy
clf_pred=clf.predict(tfidf_test)
score=accuracy_score(y_test,clf_pred)
print(f'Accuracy: {round(score*100,2)}%')


# ## PASSIVE AGGRESSIVE CLASIFIER

# In[269]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[302]:


# Initialize the PassiveAggressiveClassifier and fit training sets
pa_classifier=PassiveAggressiveClassifier(max_iter=1000,early_stopping=True,random_state=42)
pa_classifier.fit(tfidf_train,y_train)


# In[303]:


# Predict and calculate accuracy
y_pred=pa_classifier.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[272]:


# Build confusion matrix
confusion_matrix(y_test,y_pred, labels=[0,1])


# In[273]:


# OVERALL SCORES

print(f1_score(y_test, y_pred, average="macro"))
print(precision_score(y_test, y_pred, average="macro"))
print(recall_score(y_test, y_pred, average="macro"))


# In[274]:


# CLASS LEVEL SCORES FOR LABEL 0 AND LABEL 1

from sklearn.metrics import precision_recall_fscore_support as score
precision, recall, fscore, support = score(y_test, y_pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# ## RANDOM FOREST

# In[141]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[142]:


# Create and fit model 
rf_classifier = RandomForestClassifier(n_estimators = 100, random_state = 11) 
rf_classifier.fit(tfidf_train, y_train)  


# In[143]:


rf_preds = rf_classifier.predict(tfidf_test)
score=accuracy_score(y_test,rf_preds)
print(f'Accuracy: {round(score*100,2)}%')


# ## DECISION TREE

# In[144]:


from sklearn.tree import DecisionTreeClassifier


# In[145]:


# Create and fit model 
dt_classifier = DecisionTreeClassifier(criterion= 'entropy',max_depth = 20, splitter='best',random_state=42) 
dt_classifier.fit(tfidf_train, y_train)  


# In[146]:


dt_preds = dt_classifier.predict(tfidf_test)
score=accuracy_score(y_test,dt_preds)
print(f'Accuracy: {round(score*100,2)}%')


# ## TEST DATA

# FOLLOWING THE SAME PROCESS AS APPLIED ON TRAIN DATA.

# In[174]:


# Import dataset
testdf=pd.read_csv('C:\\Users\\dharmesh pathak\\Downloads\\test.csv')
# Get the shape
testdf.shape


# In[175]:


testdf


# In[177]:


testdf = testdf[['id','text']]


# In[178]:


# TOTAL NULL PRESENT IN DATA.
testdf.isnull().sum()


# In[179]:


# REMOVING ROWS WHERE TEXT IS NULL.
testdf = testdf.dropna()


# In[187]:


# PUNCTUATION REMOVAL
testdf['text'] = testdf['text'].apply(punctuation_removal)


# In[221]:


# STOPWORDS REMOVAL
testdf['text'] = testdf['text'].apply(stopwords_removal)


# In[224]:


# LEMMATIZATION OF EACH WORD IN NEWS ARTICLE.
testdf['text'] = testdf['text'].apply(lambda x: ' '.join([word.lemma_ for word in sp(x)]))


# In[225]:


testdf['text']


# In[228]:


# VECTORIZING THE TEST DATA
test_tfidf = tfidf_vectorizer.transform(testdf['text'])


# In[236]:


# CLASSIFICATION USING TRAINED PASSIVE AGGRESSIVE CLASSIFIER.
test_pred=pa_classifier.predict(test_tfidf)


# In[237]:


len(test_pred)


# In[238]:


# CREATING LABEL COLUMN HAVING PREDICTED VALUES.
testdf['label'] = test_pred


# In[242]:


submit_df = testdf.drop('text',axis=1)


# In[251]:


submit_df


# In[249]:


# WRITE submit.csv to disk
submit_df.to_csv('C:\\Users\\shubham verma\\Downloads\\submit.csv',index=False)

