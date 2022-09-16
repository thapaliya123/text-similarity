# import
import spacy
import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

# convert string into its lower case
def lower_casing(text):
    """
    Obtained the lower case version and passed lower case version of text 

    Arguments:
    text: raw text, strings

    Returns:
    lower_text: string, representing lower case of raw text
    """

    # lower casing
    lower_text = text.lower()

    return lower_text

# remove special symbols using regex
def remove_special_symbols(text):
    """
    remove specified symbols such as numbers, replace hyphen and dot with space in given text passed as an argument 
    
    Arguments: raw text, strings
    
    Returns: 
    processed_text = text obtained after preprocessing
    """
    
    # replace hyphen, dot with space
    text_with_no_hyphen = text.replace("-", " ").replace(".", " ")
    

    #specifying the things we want to remove using regular expressions and removing using sub
    regex_remove = "([0-9])|(@[A-Za-z0-9]+)|([^0-9A-Za-z t])|(w+://S+)|^RT|http.+?"
    
    clean_text = re.sub(regex_remove, '', text_with_no_hyphen).strip()
    clean_text = ''.join(clean_text)
    

    
    return clean_text
    
# remove punctuations
def remove_punctuation(text, nlp):
    """
    removes punctuation symbols present in the raw text passed as an arguments
    
    Arguments:
    text: raw text
    nlp: instantiated object called 'nlp' after loading the english model using spacy    
   
    Returns: 
    not_punctuation: list of tokens without punctuation
    """
    # passing the text to nlp and initialize an object called 'doc'
    doc = nlp(text)
    
    not_punctuation = []
    # remove the puctuation
    for token in doc:
        if token.is_punct == False:
            not_punctuation.append(token)
    
    return not_punctuation


# tokenize words
def tokenize_word(text, nlp):
    """
    Tokenize the text passed as an arguments into a list of words(tokens)
    
    Arguments:
    text: raw text
    nlp: instantiated object called 'nlp' after loading the english model using spacy    
    
    Returns:
    words: list containing tokens in text
    """
    # passing the text to nlp and initialize an object called 'doc'
    doc = nlp(text)
    
    # Tokenize the doc using token.text attribute
    words = [token.text for token in doc]
        
    # return list of tokens
    return words

# remove stopwords
def remove_stopwords(tokens, nlp):
    """
    Removes stopwords passed from the tokens list passed as an arguments
    
    Arguments:
    tokens: list of tokens
    nlp: instantiated object called 'nlp' after loading the english model using spacy    
        
    Returns:
    tokens_without_sw: list of tokens of raw text without stopwords
    """
    
    # getting list of default stop words in spaCy english model
    stopwords =nlp.Defaults.stop_words
    
    # tokenize text
    text_tokens = tokens
    
    # remove stop words:
    tokens_without_sw = [word for word in text_tokens if word not in stopwords]
    
    # return list of tokens with no stop words
    return tokens_without_sw


# Lemmatization
def lemmatization(tokens):
    """
    obtain the lemma of the each token in the token list, append to the list, and returns the list
    
    Arguments:
    text: list of tokens
    
    Returns:
    lemma_list: return list of lemma corresponding to each tokens
    """
    

    lemma_list = []
    # Lemmatization
    for token in tokens:
        lemma_list.append(token.lemma_)
    
    return lemma_list

# raw text preprocessing
def preprocess_text(text, nlp):
    """
    - preprocess raw text passed as an arguments
    - preprocessing of text includes, lower_casing, remove_special_symbols, remove_punctuation, remove_stopwords, lemmatization
    
    Arguments:
    text: raw text, string
    nlp: instantiated object called 'nlp' after loading the english model using spacy    
    Returns: list of tokens obtained after preprocessing
    """
    
    # lower casing
    lower_case_text = lower_casing(text)
    
    # remove special symbols
    removed_special_symbols = remove_special_symbols(lower_case_text)
    
    # remove punctuation
    tokens_without_punct = remove_punctuation(removed_special_symbols, nlp)
    
    # remove stopwords
    tokens_without_stopwords = remove_stopwords(tokens_without_punct, nlp)
    
    # lemmatization
    lemma_of_tokens = lemmatization(tokens_without_stopwords)

    # return preprocessed text
    return lemma_of_tokens
