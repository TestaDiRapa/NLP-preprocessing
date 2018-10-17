from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from treetaggerwrapper import make_tags, TreeTagger
import re

global tagger
tagger = TreeTagger(TAGLANG='it')

def translate_pos(pos):
    '''
    Converts a POS tag from the TreeTagger format to the WordNet format
    :parameter pos: the POS tag 
    :type pos: string
    :return: a POS tag in WordNet format
    :rtype: string
    '''
    if not isinstance(pos, str):
        raise Exception("The POS tag must be a string")
    if pos == "NOM":
        return 'n'
    elif pos[:3] == "VER":
        return 'v'
    elif pos == "ADJ":
        return 'a'
    elif pos == "ADV":
        return 'r'
    else:
        return pos

def extract_tags(text):
    '''
    Extract the POS tags from a text and lemmatizes it in Italian
    :parameter text: the text to tag
    :type text: string
    :return: a list of dictionaries with lemma and POS tag for each word
    :rtype: list
    '''
    if not isinstance(text, str):
        raise Exception("The text must be a string")
    ret = list()
    tags = tagger.tag_text(text)
    tags = make_tags(tags)
    for tag in tags:
        tmp = dict()
        tmp['word'] = tag.word
        tmp['lemma'] = tag.lemma.split('|')[0]
        tmp['pos'] = translate_pos(tag.pos.split(':')[0])
        ret.append(tmp)
    return ret

def rework(q):
    '''
    Processing operations for NLP (HTML tag removing, POS tagging, lemmatization, stopword removal)
    :param q: The text to preprocess
    :type q: string
    :return: A dictionary containing all the informations
    :rtype: dict
    '''
    tmp = dict()
    soup = BeautifulSoup(q, 'lxml')
    original = soup.get_text()
    question = re.sub("[^A-Za-zèéàùòì.,;:']", ' ', original)
    question = re.sub("http", '', question)
    question = re.sub("([.,:;])", '\g<1> ', question)
    question = re.sub (' {2,}', ' ', question)
    question = question.lower()
    question = extract_tags(question)
    question = [word for word in question if len(re.sub('[^a-zèéàùòì]', '', word['word'])) > 1 and word['word'] not in set(stopwords.words('italian'))]
    tmp['lemmas'] = [word['lemma'] for word in question]
    tmp['question'] = original
    tmp['preproc'] = q
    return tmp
