import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from sklearn.preprocessing import LabelEncoder
import warnings
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron

warnings.filterwarnings("ignore")
morph = MorphAnalyzer()
stop_words = set(stopwords.words('english'))
lemma = False
def _preprocess_text(text):
    tokens = word_tokenize(text.lower())
    # tokens =[token for token in tokens if token.isalnum() and token not in stop_words]
    if lemma:
        tokens = [morph.parse(token)[0].normal_form for token in tokens]
    return tokens

def preprocess_text_Word2Vec(text):
    return _preprocess_text(text)

def preprocess_text_Tfidf(text):
    return ' '.join(_preprocess_text(text))

def prep_data(data, preprocess_text, name_base, name_clusters):
    words = data[name_base].apply(preprocess_text)
    le = LabelEncoder()
    clusters = le.fit_transform(data[name_clusters])
    return words, clusters

def w2v(w_Word2Vec):
    Word2Vec_vectorizer = Word2Vec(w_Word2Vec, vector_size=100, window=5, min_count=1, sg=0)
    return [Word2Vec_vectorizer.wv[tokens].mean(axis=0) for tokens in w_Word2Vec]

def tfidf(w_Tfidf):
    tfidf_vectorizer = TfidfVectorizer()
    return tfidf_vectorizer.fit_transform(w_Tfidf)

def count_accuracy(vectors, c, clf):
    kmeans = KMeans(n_clusters=c)
    clusters = kmeans.fit_predict(vectors)

    X_train, X_test, y_train, y_test = train_test_split(vectors, clusters, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    return accuracy, f1 

if len(sys.argv) == 4:
    name = sys.argv[1]
    base_col_name = sys.argv[2]
    clusers_col_name = sys.argv[3]
    data = pd.read_csv(name)
    clfs = [RandomForestClassifier(), Perceptron(), MLPClassifier(activation='relu',
                                                    solver='lbfgs',
                                                    hidden_layer_sizes=(150,), 
                                                    early_stopping=True,
                                                    random_state=42,
                                                    max_iter=100)]  
    clf_names = ["Случайный лес", "Перцептрон", "Многослойный перцептрон"] 
    lemmas = [False, True]
    lemmas_names = ["нет", "да"]
    processors = [(preprocess_text_Word2Vec, w2v),
                  (preprocess_text_Tfidf, tfidf)]
    len_cols = [40, 30, 30, 15, 15]
    names = ["контекстуализированная векторная модель", "tf*idf"]

    print("x" + "-" * len_cols[0] + "x" + "-" * len_cols[1] + "x" + "-" * len_cols[2] + "x"+ "-" * len_cols[3] + "x"+ "-" * len_cols[4] + "x")
    print(('|{:^' + str(len_cols[0]) + '}|{:^' + str(len_cols[1]) + '}|{:^' + str(len_cols[2]) + '}|{:^' + str(len_cols[3]) + '}|{:^' + str(len_cols[4]) + '}|').format("метод векторизации", "морфологический анализ", "классификатор", "accuracy", "f-мера"))
    print("x" + "-" * len_cols[0] + "x" + "-" * len_cols[1] + "x" + "-" * len_cols[2] + "x"+ "-" * len_cols[3] + "x"+ "-" * len_cols[4] + "x")
    f = '|{:^' + str(len_cols[0]) + '}|{:^' + str(len_cols[1]) + '}|{:^' + str(len_cols[2]) + '}|{:^' + str(len_cols[3]) + '.3f}|{:^' + str(len_cols[4]) + '.3f}|'
    for k in range(len(clfs)):
        for j in range(len(processors)):
            for i in range(len(lemmas)):
                lemma = lemmas[i]
                words, c = prep_data(data, processors[j][0], base_col_name, clusers_col_name)
                vectors = processors[j][1](words)
                acc, f_m = count_accuracy(vectors, len(np.unique(c)), clfs[k])
                print(f.format(names[j], lemmas_names[i], clf_names[k], acc, f_m))
                print("x" + "-" * len_cols[0] + "x" + "-" * len_cols[1] + "x" + "-" * len_cols[2] + "x"+ "-" * len_cols[3] + "x"+ "-" * len_cols[4] + "x")
            
