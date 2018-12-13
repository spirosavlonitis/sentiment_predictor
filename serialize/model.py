import numpy as np
from stream_docs import stream_docs
from get_minibatch import get_minibatch
import sys 
sys.path.append("..")
from tokenizer import tokenizer
from vectorizer import vect
from sklearn.linear_model import SGDClassifier
from pyprind import ProgBar
import os
import pickle

clf = SGDClassifier(loss='log', random_state=1, max_iter=1)

classes = np.array([0,1])
doc_stream = stream_docs('movie_data.csv')
pbar = ProgBar(45)

for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, 1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream, 5000)
X_test = vect.transform(X_test)

print("Accuracy: %.3f" % clf.score(X_test, y_test))
clf.partial_fit(X_test, y_test, classes=classes)

pickle.dump(clf,
        open(os.path.join('..','pkl_objects', 'classifier.pkl'), 'wb'),
        protocol=4
    )