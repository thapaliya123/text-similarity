# import necssary packages
import spacy
import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
# import preprocess_text module from preprocessing_text.py
from preprocessing_text_spacy import preprocess_text

# load the english model and initialize an object called 'nlp'
nlp = spacy.load("en_core_web_sm")

# calulate jaccard similarity
def jaccard_similarity(text1, text2):
    """
    calculate jaccard similarity index between two preprocessed tokenized list of text i.e. text1, text2
    jaccard index = size of intersection / size of union

    Arguments:
    text1: article or text which is preprocessed and contains list of tokens, (list of tokens)
    text2: article or text which is preprocesed and contains list of tokens, (list of tokens)
    
    Returns:
    jaccard_score: result of similarity between two text using jacard index
    """
    # intersection between two text i.e. text1, text2
    intersection = set(text1).intersection(set(text2))
    
    # union between two text i.e. text1, text2
    union = set(text1).union(set(text2))    
    
    # calculate jaccard score
    jaccard_score = len(intersection)/len(union)
    

    return jaccard_score

# obtain term documents and calculate cosine similarity
def obtain_term_document(text1, text2):
    """
    Takes two preprocessed tokenized text i.e. text1 and text2 and calculate the term in each documents and return
    pandas dataframe representing frequency of union of words in text1 and text2 for each documents i.e. text1 and text2
    in normalized form
    
    Arguments:
    text1: Preprocessed tokenized text or documents, list of tokens
    text2: Preprocessed tokenized text or documents, list of tokens
    
    Returns:
    df: pandas dataframe containing frequency of word terms for each documents in normalize form
    """
    # union of word tokens in text1 and text2
    union_of_words = set(text1).union(set(text2))
    
    word_count_text1 = [] #initialeze for sotring frequency of words
    word_count_text2 = [] #initialize for storing frequency of words
    
    # word frequency count for each text
    for word in union_of_words:
        word_count_text1.append(text1.count(word))
        word_count_text2.append(text2.count(word))
    
    # create dataframe containing frequency of words for each text i.e. text1 and text2
    df = pd.DataFrame(list(zip(word_count_text1, word_count_text2)), index=union_of_words, columns=['text1', 'text2'])
    
    # normalize each text
    df["text1"] = round(df["text1"]/df["text1"].sum(), 3)
    df["text2"] = round(df["text2"]/df["text2"].sum(), 3)
    
    return df

def cosine_similarity(dataframe):
    """
    computes cosine of the angle between two vectors and helps us to find out how related two documents are
    
    Arguments:
    dataframe: dataframe containing normalized terms/words in each document
    
    Returns:
    cosine_similarity_score: cosine_similarity_score between two documents
    """
    # convert each column of dataframe which represent individual documents into array
    document_one_vector = np.array(df.iloc[:, 0]) # vector representing document1 or text1
    document_two_vector = np.array(df.iloc[:, 1]) # vector representing document2 or text2
    
    # calulate dot product of the two vector
    dot_product_of_two_vector = np.dot(document_one_vector, document_two_vector)
    
    # calculate magnitude for each vector
    document_one_vector_magnitude = np.sqrt(np.sum(np.square(document_one_vector))) # magnitude for vector document1
    document_two_vector_magnitude = np.sqrt(np.sum(np.square(document_two_vector))) # magnitude for vector document2
    
    # compute cosine similarity for two documents
    cosine_similarity_score = dot_product_of_two_vector/(document_one_vector_magnitude*document_two_vector_magnitude)
    
    return cosine_similarity_score


# sample article 1
article_1 = """Nepal on Thursday reported 12 more Covid-19-related fatalities, pushing the death toll to 1,663. 
The country also recorded 1,217 new cases.The overall infection tally has reached 245,650 with 12,386 active cases.
According to the Ministry of Health and Population, 231,601 infected people have recovered from the disease so far; 
1,064 of them in the past 24 hours."""

# sample article 2
article_2 = """As of Thursday, the number of confirmed cases in the Valley has reached 114,409 While Kathmandu has
reported 432 Covid-19-related fatalities so far, Lalitpur and Bhaktapur have recorded 129 and 97 deaths respectively."""

# preprocess the text
preprocessed_article_1 = [item for item in preprocess_text(article_1, nlp) if item not in [' ', '  ']]
preprocessed_article_2 = [item for item in preprocess_text(article_2, nlp) if item not in [' ', '  ']]

# compute jaccard similarity socre
jaccard_similarity_score = jaccard_similarity(preprocessed_article_1, preprocessed_article_2)

# compute cosine similarity score
df = obtain_term_document(preprocessed_article_1, preprocessed_article_2)
cosine_similarity_score = cosine_similarity(df)

# add jaccard_similarity_score and cosine_similarity_score to dataframe
data = {'similarity_score':[jaccard_similarity_score, cosine_similarity_score]}
similarity_score = pd.DataFrame(data, index = ['jaccard_similarity', 'cosine_similarity'])


# similarity score represented using pandas dataframe
print("similarity_score:\n", similarity_score)