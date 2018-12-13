import re
from nltk.corpus import stopwords

def tokenizer(text):
    """Return a word tokenized text."""
    stop = stopwords.words('english')
    text = re.sub("<[^>]*>", '', text) # remove HTML tags
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", text)
    text = re.sub("[\W]+", ' ', text.lower()) \
            + ' ' + ' '.join(emoticons).replace('-', '')
    return [ w for w in text.split() if w not in stop and w not in ['p', 'd'] ]