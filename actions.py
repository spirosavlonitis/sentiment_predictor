import numpy as np
from vectorizer import vect
import sqlite3


def classify(clf, document):
    """Classify user input."""
    label = ['negative', 'positive']
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return proba, label[y]

def train(clf, document, label):
    """Train model using user input."""
    X = vect.transform([document])
    clf.partial_fit(X, [label])

def update_db(document, label):
    """Add user input to review_db."""
    conn = sqlite3.connect('reviews.sqlite')
    c = conn.cursor()
    c.execute('INSERT INTO review_db (review, sentiment, date) '\
        'VALUES (?,?, DATETIME("now"))', (document, label))
    conn.commit()
    conn.close()