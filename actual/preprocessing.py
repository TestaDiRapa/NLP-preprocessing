import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from bs4 import BeautifulSoup
from pymongo import MongoClient
from treetaggerwrapper import make_tags, TreeTagger
import mysql.connector
import re

global tagger
tagger = TreeTagger(TAGLANG='it')

def translate_pos(pos):
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
    ret = list()
    tags = tagger.tag_text(question)
    tags = make_tags(tags)
    for tag in tags:
        tmp = dict()
        tmp['word'] = tag.word
        tmp['lemma'] = tag.lemma.split('|')[0]
        tmp['pos'] = translate_pos(tag.pos.split(':')[0])
        ret.append(tmp)
    return ret

cnx = mysql.connector.connect(host='localhost', database='test_questions')
cursor = cnx.cursor()

client = MongoClient('localhost', 27017)
db = client.tesi.corpus

query = "SELECT ID, question, answer, categories FROM cm_mod_pm_question WHERE categories LIKE '%,%' LIMIT 20000"
cursor.execute(query)

corpus = list()
for element in cursor.fetchall():
    tmp = dict()
    tmp["_id"] = element[0]
    tmp['answer'] = element[2]
    tmp['categories'] = element[3].split(',')
    soup = BeautifulSoup(element[1], 'lxml')
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
    tmp['preproc'] = question
    corpus.append(tmp)

for element in corpus:
    db.insert_one(element)
