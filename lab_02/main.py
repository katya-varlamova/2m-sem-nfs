import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

def read_prep_data():
    data = pd.read_csv('Speed Dating Data.csv')

    le = LabelEncoder()
    for col in data.columns:
        data[col] = le.fit_transform(data[col])

    X = data.drop('match', axis=1)
    y = data['match']
    return X, y

def research_k_neighbours(X, y):
    neighborsRange = range(2, 12, 2)
    for m in ["correlation", "euclidean", "manhattan", "chebyshev", "minkowski"]:
        vals = [precision_recall_fscore_support(y,
                                               cross_val_predict(KNeighborsClassifier(n_neighbors=i,
                                                                                      weights='distance',
                                                                                      metric = m),
                                                                 X, y, cv=5),
                                               average='weighted')[2] for i in neighborsRange]


        plt.plot(neighborsRange, vals, label = m)
    plt.xlabel('количество соседей')
    plt.ylabel('F-мера')
    plt.legend()
    plt.title(f'F-мера от количества соседей')
    plt.savefig("зависимости KNeighbors.png")
    plt.clf()
    
def research_random_forest(X, y):
    neighborsRange = range(1, 20)
    vals = [precision_recall_fscore_support(y,
                                           cross_val_predict(RandomForestClassifier(max_depth = i),
                                                             X, y, cv=5),
                                           average='weighted')[2] for i in neighborsRange]


    plt.plot(neighborsRange, vals)
    plt.xlabel('высота дерева')
    plt.ylabel('F-мера')
    plt.title(f'F-мера от высоты дерева')
    plt.savefig("зависимости случайный лес.png")
    plt.clf()


def research_cross_val(X, y):
    range_step = np.arange(2, 15)
    classifiers = {
        'KNeighbors': KNeighborsClassifier(),
        'Случайный лес': RandomForestClassifier()
    }
    for name, clf in classifiers.items():
        print(f"Классификатор: {name}")
        y_pred = cross_val_predict(clf, X, y, cv=5)
        
        cm = confusion_matrix(y, y_pred)
        print(f"Матрица ошибок:")
        print(cm)
        
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='weighted')
        print(f"F-мера: {f1}")
        
        plt.figure()
        plt.plot(range_step, [precision_recall_fscore_support(y, cross_val_predict(clf, X, y, cv=i), average='weighted')[2] for i in range_step])
        plt.xlabel('шаг кросс-валидации')
        plt.ylabel('F-мера')
        plt.title(f'F-мера от шага кросс-валидации для "{name}"')
        plt.ylim([0, 1])
        plt.savefig(f"{name}.png")
        plt.clf()

X, y = read_prep_data()
research_random_forest(X, y)
research_k_neighbours(X, y)
research_cross_val(X, y)
