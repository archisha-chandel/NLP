#!/usr/bin/env python
# coding: utf-8

# # Introduction to Topic Modelling

# ## Description
# 
# Topic Modelling is all about extracting **topics** from a corpus of documents.  There are numerous methods to do this, however the end goal is the same. Determining what a particular document is about. Topic Modelling and Topic Classification are two entirely different aspects of the same problem and their methodologies and intuition differ considerably. We will learn the differences between a generative model and a discriminative model. And how to implement each of them. We wil stick to the earlier data, the **Consumer Complaints Database** and this time instead of trying to predict the category of the complaint we will attempt to figure out the topics the complaint are all about. 
# 
# ## Overview
# 
# - Introduction to the problem statement **Consumer Complaints Database- Categorize complaints into categories**
# - What is Topic Modelling
# - What is Topic Classification
# - Differences between a generative and a discriminative model
# - Named Entity Recognition and introduction to textBlob
# - Introduction to LSI
# - Introduction to LDA and Gensim
# - Creating Bigram and Trigram Models
# - Create the Dictionary and Corpus needed for Topic Modeling
# - Building the Topic Model
# - View the topics in LDA model
# - Compute Model Perplexity and Coherence Score
# - Building LDA Mallet Model
# - How to find the optimal number of topics for LDA?
# - Finding the dominant topic in each sentence
# - Find the most representative document for each topic
# - Topic distribution across documents
# - T-Sne and UMAP
# 
# 
# ## Pre-requisite
# 
# - Python (along with NumPy and pandas libraries)
# - Basic statistics (knowledge of central tendancy)
# - Basics of NLP
# 
# 
# ## Learning Outcomes
# 
# - Understanding the differences between Topic Modelling and Topic Classification
# - LDA, Use Cases, Coherence scores
# - Introduction to Gensim and Spacy in addition to Sklearn
# - NMF

# ## Chapter 1: Introduction to the problem statement
# 
# ### 1.1 Introduction to the problem statement: <font color='green'> Categorize complaints into categories</font>
# 
# **What is the problem?**
# We have covered the problem statement in the earlier tutorial. The data remains the same. However the problem statement is different. Earlier, the idea was to predict the complaint category, from each of the compliants listed in the column, consumer complaint narrative. Now, we have to figure the various **topics** the complaints are all about. The topics could be anything the complaints are about; banks, loans, defaults, etc; 
# 
# [https://onedrive.live.com/download?cid=C132C52D965EBCB9&resid=C132C52D965EBCB9%21689&authkey=AJJrYBK5jOE8Rw4] 
# 
# Hence, this time, the column of interest is just the consumer complaint narrative. The rest are irrelevant as of now, as we are just intereted in knowing the topics the complaints are all about, and that is described in detail in the consumer complaint narrative column. 
# 
# - Consumer Complaint Narrative
# 
# 
# **Brief explanation of the dataset & features**
# 
# * `Consumer Complaint Narrative`: Is a paragraph (or text) written by the customer explianing his complaint in detail. It is not a numerical or categorical type, the data is a string type consisting of text in the form of paragraphs
#     
# 
#  
# **What we want as the outcome?**
# 
# The various topics the complaints are about. 
# 
# ### 1.2 What is Topic modelling?
# 
# ***
# 
# **Intuition for text**
# 
# Topic Modeling is a technique to extract the hidden topics from large volumes of text.
# The challenge, however, is how to extract good quality of topics that are clear, segregated and meaningful. This depends heavily on the quality of text preprocessing and the strategy of finding the optimal number of topics. This tutorial attempts to tackle both of these problems.
# 
# **Why NLP for this data**
# 
# In the last tutorial we covered the basic cleaning of text data - removing stopwords, lemmatization, etc; Cleaning is imperative in topic modelling or topic classification as crud (or garbage data) could offset the topic modelling massively and give completely different results. 
# 
# ### Have a look at the data set 
# 
# In this task you will load Consumer_complaints.csv into a dataframe using pandas and explore the column Consumer Complaint Narrative.
# 
# 
# ### Instructions
# - Load the csv into a dataframe
# - Drop all columns except the consumer Complaints Narratives. Make sure to keep a copy of the original dataframe in a different instance. 
# - Print out the first 5 instances of our 2 column dataframe. Name it df. 
# - Rename column Consumer compliant narrative to X. 
# - print out the first value of the X column

# In[1]:


import pandas as pd
df = pd.read_csv("new_complaints.csv")
df_copy = df.copy()
df = df[["Consumer complaint narrative", "Product"]] #keeping the relevant columns
df.columns = ["X","y"]
df.head()
#Printing out the first non-empty value of the X column. Hence the second value, index is 1
print(df["X"][1]) 


# ## Chapter 2: What is Topic Classification?
# 
# ### 2.1 Introduction
# 
# ***
# 
# Topic Classification - and Topic Modelling are two different aspects of the same problem. To figure out the topics from a particular snippet of text data. End result being the same, Topic modelling and Topic classification are different ends of the spectrum converging to the same midpoint. 
# 
# Topic classification is simply put, finding the different topics text is about. This is achieved either by a simple counter function finding the different counts of the various words in the text and then figuring out what these words are and assign the document to these words **as topics**. 
# 
# The problem with this approach however being that, not all words are topics. The word, "advice" might be one of the top words in the document, doesn't necessarily imply that the document is about advice. A brilliant yet impeccably simple way to get around this is to use something called NER or Named Entity Recognition, finding out the Places, Animals, People or **Named Entities** the document is about and then conclude the document is about the highest appearing NERs. We will get to the core basics of NER eventually. 
# 
# Another additional method we use is called POS or **Parts of Speech Tagging**, finding out the various aspects of the document, such as Nouns, Verbs, etc; So we know that the word food has appeared multupl times, and it being a noun, the document is about food and an additional verb such as "reviewing". The document is about reveweing food!
# 
# 
# 
# ### 2.2 Why is it important to determine topics
# 
# ***
# 
# Knowing what people are talking about and understanding their problems and opinions is highly valuable to businesses, administrators, political campaigns. And it’s really hard to manually read through such large volumes and compile the topics.
# 
# Thus is required an automated algorithm that can read through the text documents and automatically output the topics discussed.
# 
# We will deal with each of the above concepts in detail below. 

# ## Chapter 3: What is Topic Modelling?
# 
# ### 3.1 Introduction - The difference between a generative model and a discriminative model. 
# 
# ***
# 
# Topic Modelling unlike Topic classification assigns random topics to a document **before parsing the document!**
# 
# In topic classification, the topics are found after finding the words present in the document, Topic modelling, topics which are essentially combination of words (not single words alone) are assigned to the document beforehand and then the probabilty of those topics belonging to the document is determined. The word-combos with the highest probability are assigned to the document as the topics. 
# 
# 
# Topic modelling is a generative model as the number of topics are decided beforehand and not decided after finding the most commonly occuring words in the document, in other words, any word-combo can be a topic, based on words around it, The probabilities are then assigned of those word-combinations belong to that document based on occurence and thee topics determined. Starts with the generation of the words first and then tries to fit the document to those words. 
# 
# Topic Classification is a discriminative model, as it simply discrimates the topics **present** in the document. The words are determined based on their presence in the document.

# ## Chapter 4: Getting started with Topic classification and Topic modelling 
# 
# ### 4.1 Introduction - Topic Classfication
# 
# ***
# 
# Topic modeling is the process of discovering groups of co-occurring words in text documents. These group co-occurring related words makes "topics".
# 
# LSA or Latent Semantic analysis (also referred to as LSI- Latent Semantic Index) is a Topic modelling technique. Conceptually, similar to the the SVD, LSA facilitates a term-document matrix, which is then broken down to determine the **topics** or **combination of terms** in the documents, which are the supposed topics. LSA is typically a PCA technique. 
# 
# Before we discuss SVD, Term Document matrices and the rest, let us start from scratch and do a normal topic classification on the first five non empty complaints of the comsumer complaints database.
# 

# ### Instructions
# - Drop all empty rows from df
# - Take the top 5 rows of the column X and lowercase and assign them to a list called BoW
# - Create a stopword and punctuation list (like the last tutorial)
# - Tokenize BoW into constituent words and remove stop words
# - Lemmatize using Wordnet lemmatizer and assign the list back to BoW

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


# Now that we have the bag of words, we need to figure out the topics from the same. A discriminative model, where we deduce the topics from the bag of words, based on what is present and what is not. One way to do that is NER or Named Entity Recognition...

# ### 4.2 Named Entity recognition and introduction to TextBlob
# ***
# 
# NER typically identifies, entities such as animals, places, things etc; from text. Evidently, that will be massively helpful in determining which entities are present in the bag of words, hence zeroing in on the topics. Textblob is a library like NLTK. Textblob has certain useful functionalities such as tags, tokenize, polarity analysis etc; however, we will stick to just using textblob to find the Named Entities for now. 
# 
# Textblob needs a string to work with and doesn't work with tokenized sentences. Hence it makes sense to convert all of the words in our BoW list, and join them into one string using the join command. Let's try that and see what we get. 

# In[3]:


from textblob import TextBlob as tb
BoW_joined= " ".join(BoW)
blob = tb(BoW_joined)
blob.tags[:10]


# In[4]:


len(BoW)


# Our original BoW had 602 words. Which means for each of the words we will have a tag. The various tags such as "NN", "DT" have a more detailed explanation up here, what each tag stands for. Expanding that list below we see that the tags are such:
# 
# The entire list can be found here: https://www.clips.uantwerpen.be/pages/mbsp-tags

# <img src="tag.png">

# Here's the problem. We get 602 tags. We need a smaller number. We cannot say the first 5 documents are about 602 topics which is basically about all the words combined. One way to do this would be to determine the tags of the top 5 or 10 most frequent words in this unique list and then take the tags for those words. 

# In[5]:


from collections import Counter
d = dict(Counter(BoW))
import operator
sorted_d = sorted(d.items(), key=operator.itemgetter(1), reverse=True)


# In[6]:


sorted_d[:10]

#And the top 10 words are as below


# The topics of the documents are about
# 
# - Payment
# - Closing
# - account
# - Credit
# - Mortgage
# - Rate
# - would
# - bank
# - fixed

# However words like "would" are not important as they do not really imply a topic, which means we will have to repeat this exercise for the **Nouns** only. Why? **Because the nouns usually are the places, names of people, banks etc; **

# ### Instructions
# - Get the tags of all the words from BoW, that has already been done as blob.tags in an earlier exercise
# - Extract the ones which are nouns only ("NN") to a list called nouns
# - The words which have appeared most frequently, the top 10 words are taken into a list called top
# - We see if the top have any words which are part of the nouns list.
# - Print those words

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


# The top 6 topics of the 5 complaints are now:
# - payment
# - credit
# - account
# - rate
# - mortage
# - bank

# That's topic classification 101. 

# ### 4.3 Topic Modelling and introduction to LSI
# 
# ***
# 
# The first complaint which looks like this, the topics can be inferred by reading through. 

# <img src="first.png">

# <img src="1.png">

# Words such as Loans, credit scores, credit bureau all are financial terms and can be classified collectively as Topic 1, vehicle is a different topic from these words, company is a third topic and so on. A good topic model will identify similar words and put them under one group or topic. The most dominant topic in the above example is Topic 1, which indicates that this piece of text is primarily about loans and credit
# 
# 

# Topic modelling in a way is an unsupervised technique, because, unlike topic classification which just involves figuring out what the words that are contained in the document and assigning them as topics, in topic modelling, the topics are words which are related to each other (The topics are combination of words). Hence multiple words together form a topic, and multiple topics can exist in a document. 
# 
# Topic modeling helps in exploring large amounts of text data, finding clusters of words, similarity between documents, and discovering abstract topics. Topic classification just maps words to documents, whereas topic modelling attempts to discover underlying words and combinations of words which are not visible doing just a mere preliminary analysis of the words contained. These underlying combinations are abstract or hidden and have to be figured out by running specific NLP techniques, hence the "Latent" in LSA. 

# An example should make this clearer. 
# 
# "He was asked to close the case"
# "He is close to his friend". 
# 
# The word close used in very different contexts, just a normal topic classification, wouldn't help in determining the context of the word "close". To determine the context, we will have to see the words used around the word of interest and assign these combinations as topics. For instance, close - Case is a topic 1 and close - friend is topic 2 in this case. These combinations are not visible inherenetly, unless you capture the context around the word of interest and that is Topic modelling attempts to do. 

# LSA is a topic modelling technique, hence the number of topics have to determined beforehand and then assigned to the documents.For m documents, you will assign k topics to be extracted out of these documents. 
# 
# k = number of topics we intend to extract
# 
# m = number of documents
# 
# n = number of unique words in all the documents combined
# 
# So if we had a word **X** document matrix, the values for the words being the TF-IDF scores for each word depending on how many times it has appeared in the document and across all documents, it would look like below. 

# <img src="2.png">

# To extract K topics out of this MXN matrix, we will have to break it down into 2 matrices, M X K and K X N the product of which would be M X N or our original matrix. This makes sense as the rows and columns of 2 matrices determine the shape of the product matrix and that is possible if and only if the number of columns of matrix 1 is equal to the number of rows of matrix 2. 

# Now that we understand the math behind LSA, let's implement LSA in python with the first 5 complaints yet again. 

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


# Now to build our LSA. We need to extract 5 topics hence n_components is 5. 

# In[10]:


from sklearn.decomposition import TruncatedSVD
svd_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(vector_values_array)
len(svd_model.components_)


# The components of svd_model are our topics, and we can access them using svd_model.components_. Finally, let’s print a few most important words in each of the 5 topics and see how our model has done.

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


# ### 4.4 Introduction to LDA and Gensim
# 
# ***
# 
# LDA stands for Latent Dirichlet Allocation. 
# 
# LDA’s approach to topic modeling is it considers each document as a collection of topics in a certain proportion. And each topic as a collection of keywords, again, in a certain proportion. \
# 
# LSI learns latent topics by performing a matrix decomposition (SVD) on the term-document matrix.
# 
# LDA is a generative probabilistic model, that assumes a Dirichlet prior over the latent topics.
# 
# Briefly put, it answers the question: “given this type of distribution, what are some actual probability distributions I am likely to see?”
# 
# Consider the very relevant example of comparing probability distributions of topic mixtures. Let’s say the corpus we are looking at has documents from 3 very different subject areas. If we want to model this, the type of distribution we want will be one that very heavily weights one specific topic, and doesn’t give much weight to the rest at all. If we have 3 topics, then some specific probability distributions we’d likely see are:
# 
# Mixture X: 90% topic A, 5% topic B, 5% topic C
# Mixture Y: 5% topic A, 90% topic B, 5% topic C
# Mixture Z: 5% topic A, 5% topic B, 90% topic C
# 
# 
# LDA assumes documents are produced from a mixture of topics. Those topics then generate words based on their probability distribution. Given a dataset of documents, LDA backtracks and tries to figure out what topics would create those documents in the first place.
# 
# 
# Once you provide the algorithm with the number of topics, all it does it to rearrange the topics distribution within the documents and keywords distribution within the topics to obtain a good composition of topic-keywords distribution.
# 
# Again, When I say topic, what is it actually and how it is represented?
# 
# A topic is nothing but a collection of dominant keywords that are typical representatives. Just by looking at the keywords, you can identify what the topic is all about.
# 
# Just like the LSA, It Iterates through each word “w” for each document “d” and tries to adjust the current topic – word assignment with a new assignment. A new topic “k” is assigned to word “w” with a probability P which is a product of two probabilities p1 and p2.
# 
# For every topic, two probabilities p1 and p2 are calculated. P1 – p(topic t / document d) = the proportion of words in document d that are currently assigned to topic t. P2 – p(word w / topic t) = the proportion of assignments to topic t over all documents that come from this word w.
# 
# The current topic – word assignment is updated with a new topic with the probability, product of p1 and p2 . In this step, the model assumes that all the existing word – topic assignments except the current word are correct. This is essentially the probability that topic t generated word w, so it makes sense to adjust the current word’s topic with new probability.
# 
# After a number of iterations, a steady state is achieved where the document topic and topic term distributions are fairly good. This is the convergence point of LDA.

# **Importing the necessary packages**

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


# Topic 0 is a represented as 0.028“closing” + 0.021“account” + 0.016“rate” +.... and so on.
# 
# It means the top 7 keywords that contribute to this topic are: ‘closing’, ‘xxxx’, ‘account’, 'rate'.. and so on and the weight of ‘closing’ on topic 0 is 0.028.
# 
# The weights reflect how important a keyword is to that topic.
# 
# Looking at these keywords, can you guess what this topic could be? You may summarise it to ‘closing’ or 'account'.
# 
# Likewise, can you go through the remaining topic keywords and judge what the topic is?

# A better way to display this would be like so. 

# In[31]:


pprint(ldamodel.print_topics())
doc_lda = ldamodel[doc_term_matrix]


# Model perplexity and topic coherence provide a convenient measure to judge how good a given topic model is. Lower the better.

# In[32]:


# Compute Perplexity
print('\nPerplexity: ', ldamodel.log_perplexity(doc_term_matrix))  
# a measure of how good the model is. lower the better.


# In[33]:


# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=ldamodel, texts=doc_clean, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# Now that the LDA model is built, the next step is to examine the produced topics and the associated keywords. There is no better tool than pyLDAvis package’s interactive chart and is designed to work well with jupyter notebooks.

# ### 4.5 Visualize the topics
# 
# ***

# In[35]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)
vis


# <img src="vis.png">

# ### 4.6 Building the LDA Mallet model
# 
# ***
# So far you have seen Gensim’s inbuilt version of the LDA algorithm. Mallet’s version, however, often gives a better quality of topics.
# 
# Gensim provides a wrapper to implement Mallet’s LDA from within Gensim itself. You only need to download the zipfile, unzip it and provide the path to mallet in the unzipped directory to gensim.models.wrappers.LdaMallet.

# In[36]:


get_ipython().system('curl "https://onedrive.live.com/download?cid=C132C52D965EBCB9&resid=C132C52D965EBCB9%21698&authkey=AFL6D_Ewsc0C3D4" -L -o mallet1.zip')


# In[37]:


import os


# In[38]:


# mallet_path = '/' # update this path
# ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=doc_term_matrix, num_topics=20, id2word=dictionary)


# ### 4.6 How to find the optimal number of topics for LDA?
# 
# ***
# 
# Finding the optimal number of topics is to build many LDA models with different values of number of topics (k) and pick the one that gives the highest coherence value.
# 
# Choosing a ‘k’ that marks the end of a rapid growth of topic coherence usually offers meaningful and interpretable topics. Picking an even higher value can sometimes provide more granular sub-topics.
# 
# If you see the same keywords being repeated in multiple topics, it’s probably a sign that the ‘k’ is too large.
# 
# The compute_coherence_values() (see below) trains multiple LDA models and provides the models and their corresponding coherence scores.

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


# ### 4.6 Finding the dominant topic in each sentence
# 
# ***
# 
# 
# 
# One of the practical application of topic modeling is to determine what topic a given document is about.
# 
# To find that, we find the topic number that has the highest percentage contribution in that document.
# 
# The format_topics_sentences() function below nicely aggregates this information in a presentable table.

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
### 4.6 Finding the dominant topic in each sentence
df_dominant_topic.head(10)


# ### 4.6 Find the most representative document for each topic
# 
# ***
# 
# 
# Sometimes just the topic keywords may not be enough to make sense of what a topic is about. So, to help with understanding the topic, you can find the documents a given topic has contributed to the most and infer the topic by reading that document.

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


# ### 4.6 Topic distribution across documents
# 
# ***
# 
# Finally, we want to understand the volume and distribution of topics in order to judge how widely it was discussed. The below table exposes that information.

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


# In[ ]:





# ### 4.6 Topic visualizations
# 
# ***
# 

# In[ ]:




