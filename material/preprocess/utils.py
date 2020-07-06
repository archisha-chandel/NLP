import re
import string
import nltk

import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer

def process_text(text):
    """
    Process text function. Creates a list of processed vocabulary tokens by
    - converting to lower case
    - removing stock market tickers, old style re-tweets 
    - stemming words
    - removing stopwords, punctuations, digits, #, hyperlinks
    
    Input:
        tweet: a string containing a text
    Output:
        tweets_clean: a list of words containing the processed text

    """
    
    nltk.download('stopwords')
    
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    text = re.sub(r'\$\w*', '', text)
    # remove old style retweet text "RT"
    text = re.sub(r'^RT[\s]+', '', text)
    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
    # remove hashtags
    # only removing the hash # sign from the word
    text = re.sub(r'#', '', text)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    text_tokens = tokenizer.tokenize(text)
    # remove digits
    text_tokens = [t for t in text_tokens if not any(c.isdigit() for c in t)] 

    text_clean = []
    for word in text_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            text_clean.append(stem_word)

    return text_clean

if __name__ == "__main__":
    process_text()