# ## Description
# 
# Topic Modelling is all about extracting **topics** from a corpus of documents.  There are numerous methods to do this, however the end goal is the same. Determining what a particular document is about. Topic Modelling and Topic Classification are two entirely different aspects of the same problem and their methodologies and intuition differ considerably. We will learn the differences between a generative model and a discriminative model. And how to implement each of them. We wil stick to the earlier data, the **Consumer Complaints Database** and this time instead of trying to predict the category of the complaint we will attempt to figure out the topics the complaint are all about. 


# In[1]:


import pandas as pd
df = pd.read_csv("new_complaints.csv")
df_copy = df.copy()
df = df[["Consumer complaint narrative", "Product"]] #keeping the relevant columns
df.columns = ["X","y"]
df.head()
#Printing out the first non-empty value of the X column. Hence the second value, index is 1
print(df["X"][1]) 



# In[43]:


df = df.dropna()
first_five_complaints = df.head()
BoW = first_five_complaints["X"].str.lower().tolist()
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from string import punctuation
custom = list(set(stopwords.words('english')))+list(punctuation)+['``', "'s", "...", "n't"]
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
BoW = [word_tokenize(x) for x in BoW]
BoW = [item for sublist in BoW for item in sublist]
BoW = [x for x in BoW if x not in custom]
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import nltk
nltk.download('wordnet')
BoW = [lemmatizer.lemmatize(x) for x in BoW]


# In[45]:

len(BoW)

# In[3]:


from textblob import TextBlob as tb
BoW_joined= " ".join(BoW)
blob = tb(BoW_joined)
blob.tags[:10]


# In[4]:


len(BoW)

# In[5]:


from collections import Counter
d = dict(Counter(BoW))
import operator
sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)


# In[6]:


sorted_d[:10]


# In[7]:


import nltk
nltk.download('brown')
tags = blob.tags
nouns = []
for x in tags:
    if x[1]=="NN":
        nouns.append(x[0])
top_words = sorted_d[:10]
top=[]
for x in top_words:
    top.append(x[0])
top_nouns=[x for x in nouns if x in top]
top_nouns  = list(set(top_nouns))
top_nouns


# In[8]:


first_five_complaints


# In[9]:


from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents called sents
# create the transform
vectorizer = TfidfVectorizer()
all_text = first_five_complaints["X"]
all_text = pd.DataFrame(all_text)
all_text.columns = ["Text"]
all_text["Text"] = all_text['Text'].str.lower()
tfidf = TfidfVectorizer(stop_words="english")
tfidf.fit(all_text["Text"])
vector = tfidf.transform(all_text["Text"])
vector_values_array = vector.toarray()
pd.DataFrame(vector_values_array) #This below essentially is our Document - WoRD matrix




# In[10]:


from sklearn.decomposition import TruncatedSVD
svd_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(vector_values_array)
len(svd_model.components_)


# In[11]:


terms = tfidf.get_feature_names()
topics = []
for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
    topics.append("Topic "+str(i)+": ")
    for t in sorted_terms:
        topics.append(t[0])
final_topic_list = [topics[i:i+8] for i in range(0, len(topics), 8)]
for x in final_topic_list:
    print (x)


# In[12]:


# import matplotlib.pyplot as plt
# import umap.umap_ as umap

# X_topics = svd_model.fit_transform(vector_values_array)
# embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(X_topics)

# plt.figure(figsize=(7,5))
# plt.scatter(embedding[:, 0], embedding[:, 1], 
# c = dataset.target,
# s = 10, # size
# edgecolor='none'
# )
# plt.show()





# In[24]:


get_ipython().system('pip install spacy')
get_ipython().system('pip install pyLDAvis')
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


# In[27]:


list_of_docs = first_five_complaints["X"].tolist()
len(list_of_docs)
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in list_of_docs] 


# In[28]:


get_ipython().system('pip install gensim')
import gensim
from gensim import corpora
dictionary = corpora.Dictionary(doc_clean)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


# If you want to see what word a given id corresponds to, pass the id as a key to the dictionary.

# In[29]:


print ([[(dictionary[id], freq) for id, freq in cp] for cp in doc_term_matrix[:1][:10]])


# In[30]:


Lda = gensim.models.ldamodel.LdaModel
ldamodel = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=50)
print(ldamodel.print_topics(num_topics=5, num_words=7))


# In[31]:


pprint(ldamodel.print_topics())
doc_lda = ldamodel[doc_term_matrix]



# In[32]:


# Compute Perplexity
print('\nPerplexity: ', ldamodel.log_perplexity(doc_term_matrix))  
# a measure of how good the model is. lower the better.


# In[33]:


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=ldamodel, texts=doc_clean, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# ### 4.5 Visualize the topics
# 
# ***

# In[35]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)
vis


# In[36]:


get_ipython().system('curl "https://onedrive.live.com/download?cid=C132C52D965EBCB9&resid=C132C52D965EBCB9%21698&authkey=AFL6D_Ewsc0C3D4" -L -o mallet1.zip')


# In[37]:


import os


# In[38]:


# mallet_path = '/' # update this path
# ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=doc_term_matrix, num_topics=20, id2word=dictionary)


# In[39]:


import matplotlib.pyplot as plt
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        Lda = gensim.models.ldamodel.LdaModel
        model = Lda(doc_term_matrix, num_topics=5, id2word = dictionary, passes=50)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[40]:


# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=doc_term_matrix, texts=doc_clean, start=2, limit=40, step=6)


# In[41]:


limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[42]:


for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# If the coherence score seems to keep increasing, it may make better sense to pick the model that gave the highest CV before flattening out. This is exactly the case here. I will pick the one with 32 topics

# In[ ]:


optimal_model = model_list[-2]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# In[ ]:


def format_topics_sentences(ldamodel=ldamodel, corpus=doc_term_matrix, texts=doc_clean):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=doc_term_matrix, texts=doc_clean)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)

# In[ ]:


# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet.head()

# In[ ]:


# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics
