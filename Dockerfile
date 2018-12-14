FROM python:3.7

LABEL "maintainer"="<spirosavlonitis1984@gmail.com>"
LABEL "app-name"="sentiment predictor"

USER root

ENV APP /data/sentiment_predictor

RUN apt-get update -y

ADD . $APP/
WORKDIR $APP

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords

CMD ["python", "app.py"]