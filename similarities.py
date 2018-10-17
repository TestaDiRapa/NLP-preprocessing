from nltk.corpus import wordnet as wn

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity
from math import sqrt
    
global dictionary
global model
dictionary = None
model = None

def load_dictionary(path):
    '''
    Loads a dictionary from file for Gensim models
    :param path: the path
    :type path: string
    '''
    dictionary = Dictionary.load(path)

def load_tfidf_model(path):
    '''
    Loads a TF-IDF model from file
    :param path: the path
    :type path: string
    '''
    model = TfidfModel.load(path)
   

def tfidf_statistic_similarity(question1, question2):
    '''
    Calculate the TF-IDF similarity between 2 questions
    :param question1: the first question (list of strings)
    :param question2: the second question (list of strings)
    :return: the similarity index
    :rtype: float
    '''
    
    if model is None:
        raise Exception("Please set the model")
    
    if dictionary is None:
        raise Exception("Please set the dictionary")
    
    q1 = [dictionary.doc2bow(question1)]
    q2 = [dictionary.doc2bow(question2)]
    try:
        index = MatrixSimilarity(model[q1])
        sims = index[q2]
    except:
        index = MatrixSimilarity(model[q2])
        sims = index[q1]
    return sims[0][0]

def cosine_product(vect1, vect2):
    '''
    Evaluate the cosine product between 2 vectors (normalized by their length). 
    The 2 vectors must be float vectors
    :param vect1: the first vector
    :param vect2: the second vector
    :return: the cosine product
    :rtype: float
    '''    
    norm1 = len(vect1)
    norm2 = len(vect2)
    ssum = 0
    for i in range(norm1):
        ssum += vect1[i]*vect2[i]
    
    return ssum/sqrt(norm1)/sqrt(norm2)

def statistic_similarity(question1, question2):
    '''
    Evaluate the statistic similarity using a low dimensionality bag of words model.
    :param question1: a list of the words of the first question
    :param question2: a list of the words of the second question
    :return: the BoW similarity
    :rtype: float
    '''
    words1 = set()
    words2 = set()
    total = set()
    for word in question1:
        words1.add(word)
        total.add(word)
    
    for word in question2:
        words2.add(word)
        total.add(word)
        
    vect1 = list()
    vect2 = list()
    for word in list(total):
        if word in words1:
            vect1.append(1)
        else:
            vect1.append(0)
            
        if word in words2:
            vect2.append(1)
        else:
            vect2.append(0)
            
    return cosine_product(vect1, vect2)

def path_similarity(term1, term2):
    '''
    Evaluates the path similarity between two italian words using WordNet. 
    :param term1: The first word. Must be a dictionary with the following keys: "lemma": the lemma, "pos": the pos tag
    :param term2: The first word. Must be a dictionary with the following keys: "lemma": the lemma, "pos": the pos tag
    :return: the path similarity
    :rtype: float
    '''    
    if term1['pos'] in ['r', 'n', 'a', 'v']:
        lemma1 = wn.lemmas(term1['lemma'], lang='ita', pos=term1['pos'])
    else:
        lemma1 = wn.lemmas(term1['lemma'], lang='ita')

    if term2['pos'] in ['r', 'n', 'a', 'v']:
        lemma2 = wn.lemmas(term2['lemma'], lang='ita', pos=term2['pos'])
    else:
        lemma2 = wn.lemmas(term2['lemma'], lang='ita')

    if (len(lemma2) == 0 or len(lemma1) == 0) and term1['lemma'] == term2['lemma']:
        return 1
    if len(lemma1) == 0 or len(lemma2) == 0:
        return 0
    
    synset1 = lemma1[0].synset()
    synset2 = lemma2[0].synset()
	
    tmp = synset1.path_similarity(synset2)
    if tmp is None:
        return 0
    else:
        return tmp

def maxssim(word, question):
    '''
    Returns the maximum path similarity between a word and the words of a question.
    :param word: a dictionary
    :param question: a list of dictionaries
    The dictionaries format must be the one contained in the "preproc" one of the "rework" function
    of the preprocess module
    :return: the max similarity
    :rtype: float
    '''
    res = [0]
    for w in question:
        tmp = path_similarity(word, w)
        res.append(tmp)
    return max(res)

def semantic_similarity(question1, question2):
    '''
    Returns the semantic simlarity between two questions. The two questons are list of dictionaries.
    The dictionaries format must be the one contained in the "preproc" one of the "rework" function
    of the preprocess module
    :param question1: the first question
    :param question2: the second question
    :return: the semantic similarity
    :rtype: float
    '''
    sum1 = 0
    for word in question1:
        sum1 += maxssim(word, question2)
    sum1 /= len(question1)
    
    sum2 = 0
    for word in question2:
        sum2 += maxssim(word, question1)
    sum2 /= len(question2)
    
    return (sum1+sum2)/2
