#!/usr/bin/env python
# coding: utf-8

# In[54]:


#we can use pickle file as well
#train.to_pickle("train.pkl")
# test.to_csv("test.csv")


# In[2]:


import pandas as pd


# In[3]:


train = pd.read_csv("train.csv")
train.head()


# In[4]:


test = pd.read_csv("test.csv")
test.head()


# In[5]:


train_pos = train[train["sentiment"]=="pos"][:5000]
train_neg = train[train["sentiment"]=="neg"][:5000]
test_neg = test[test["sentiment"]=="neg"][:5000]
test_pos = test[test["sentiment"]=="pos"][:5000]
train = pd.concat([train_neg,train_pos])
test = pd.concat([test_neg,test_pos])


# In[6]:


train = pd.concat([train_neg,train_pos])
test = pd.concat([test_neg,test_pos])


# In[7]:


test["sentiment"].value_counts()


# In[8]:


import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from nltk.corpus import stopwords
from string import punctuation
nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('english')

custom = stop_words+list(punctuation)
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
import warnings
warnings.filterwarnings("ignore")
def my_tokenizer(s):
    s = str(s)
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t)>2] #remove words lesser than 2 in length
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] #lemmatize words
    tokens = [t for t in tokens if t not in custom] #remove stopwords and punctuation
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove digits
    return tokens


# In[9]:


text = train["train"].tolist()+test["test"].tolist()
len(text)


# In[10]:


final_text = []
count = 0
for x in text:
    
    final_text.append(my_tokenizer(x))
    count+=1
    print (count)


# In[11]:


final_text = [' '.join(x) for x in final_text]


# In[12]:


df = pd.DataFrame(final_text, columns = ["text"])


# In[13]:


df["sentiment"] = train["sentiment"].tolist()+test["sentiment"].tolist()


# In[14]:


df.sentiment.value_counts()


# In[15]:


X = df["text"]
y= df["sentiment"]


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words="english")


# In[17]:


X = tfidf.fit_transform(X)


# In[18]:


import numpy as np
X = X.toarray()


# In[19]:


# X[0].shape


# In[20]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[21]:


y = le.fit_transform(y)


# In[22]:


y


# In[23]:


type(y)


# In[24]:


X_train = X[:10000]
X_test = X[10000:]
y_train = y[:10000]
y_test = y[10000:]


# In[25]:


X_test.shape


# In[26]:


X_train.shape


# In[27]:


y_train.shape


# In[28]:


y_test.shape


# In[29]:


from sklearn.naive_bayes import GaussianNB


# In[30]:


nb = GaussianNB()


# In[31]:


from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix


# In[33]:


nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)
print (classification_report(y_test,y_pred))


# In[ ]:




