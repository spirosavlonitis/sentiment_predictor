# sentiment_predictor
A web app that predicts the sentiment of a movie review.

To run this app if you have docker installed then simply build an image, and once it's completed run it:
```shell
 $ docker build -t sentiment:latest .
 $ docker run -p 5000:5000 sentiment
```
If you wish to run it  without using docker then first you will need to install all the required libraries:
```shell
 $ pip install -r requirements.txt
```
The download stopwords for the nltk module:
```shell
$ python
>>> import nltk
>>> nltk.download('stopwords')
```

When done run the app.py script:
```shell
$ python app.py
```
