from flask import Flask, render_template , request
from wtforms import Form, TextAreaField, validators
from actions import classify, train, update_db
import pickle
import os

cur_dir = os.path.dirname(__file__)
clf = pickle.load(
        open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb')
    )

app = Flask(__name__)

class RequestForm(Form):
    moviereview = TextAreaField('Submit your movie review:',
                    [validators.DataRequired(),
                     validators.length(min=15)])

@app.route('/')
def index():
    """Home page."""
    form = RequestForm(request.form)
    return render_template('index.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    """Classify user input and render the results."""
    form = RequestForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['moviereview']
        proba, y = classify(clf, review)
        return render_template('results.html',
                                content=review,
                                prediction=y,
                                probability=round(proba*100, 2)
                                )
    return render_template('index.html', form=form)

@app.route('/thanks', methods=['POST'])
def thanks():
    """Store user feedback and update model."""
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']
    inv_label = {'negative': 0, 'positive':1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not y)      # invert prediction
    train(clf, review, y)
    update_db(review, y)
    return render_template("thanks.html")


if __name__ == '__main__':
    app.run(debug=True)