from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.test.corpus
db2 = client.test.questions

corpus = list()
for doc in db.find(no_cursor_timeout=True):
    corpus.append(doc['lemmas'])

for doc in db2.find():
    corpus.append(doc['lemmas'])

dictionary = Dictionary(corpus)
dictionary.save("../tfidf/dictionary_100.dict")

doc2bow = list()
for doc in corpus:
    tmp = dictionary.doc2bow(doc)
    doc2bow.append(tmp)


model = TfidfModel(doc2bow, id2word=dictionary)
model.save("../tfidf/model_100.model")


