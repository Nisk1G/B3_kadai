from gensim.models import Word2Vec
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


s = ''

with open('assets/20220721.txt') as f:
    for line in f:
        line = line.rstrip()
        #print(line)
        s += line
#print(s)
kigo = string.punctuation
print(kigo)
kigo = kigo.replace('.','')
#kigo = kigo.replace(',','')
kigo = kigo.replace('!','')
kigo = kigo.replace('?','')
#kigo = kigo.replace(':','')
kigo = kigo.replace("'","")
print(kigo)
sentence = s.translate(str.maketrans( '', '',kigo))

sentence = sentence.replace('.', ' . ')
sentence = sentence.replace('!', ' ! ')
sentence = sentence.replace('?', ' ? ')


sentence = sentence.lower()
#print(sentence)

tokens = nltk.word_tokenize(sentence)
#print(tokens)

#tokens = [word for word in tokens if not word in set(stopwords.words("english"))]

tokens_resized  = []
one_sentence = []

for i in tokens:
    one_sentence.append(i)
    if i == '.' or i == '!' or i == '?':
        tmp = one_sentence
        #print(tmp)
        tokens_resized.append(tmp)
        #print(tokens_resized)
        #print(one_sentence)
        one_sentence = []

#print(tokens_resized)


model = Word2Vec(sentences=tokens_resized, vector_size=500, window=5, min_count=1, sg = 1)
model.wv.save_word2vec_format('assets/sample_word2vec.txt')

#似ている単語
n = 5 # 表示する個数を 5 個にする
print(model.wv.most_similar('alice', topn=n))

print(model.wv.most_similar('think', topn=n))

print(model.wv.most_similar('good', topn=n))

#単語ベクトルの減加算
print(model.wv.most_similar(positive=['she','her'], negative=['sister','dinah','rabbit'], topn=n))

