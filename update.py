import numpy as np
import sqlite3
import pickle as pkl
from vectorizer import vect
from os import path

def update_model(batch_size=10000):
    """Update model from data stored in review_db."""
    
    clf = pkl.load(
            open(path.join('pkl_objects', 'classifier.pkl'), 'rb')
        )

    conn = sqlite3.connect('reviews.sqlite')
    c = conn.cursor()
    c.execute("SELECT * FROM review_db")

    results = c.fetchmany(batch_size)
    while results:
        data = np.array(results)
        X = data[:, 0]
        y = data[:, 1].astype(int)
        
        classes = np.array([0, 1])
        X_train = vect.transform(X)
        clf.partial_fit(X_train, y, classes=classes)
        results = c.fetchmany(batch_size)
    conn.close()
    pkl.dump(clf,
            open(path.join('pkl_objects', 'classifier.pkl'), 'wb'),
            protocol=4
        )
    return None


if __name__ == '__main__':
    update_model()