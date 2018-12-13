from tokenizer import tokenizer
from sklearn.feature_extraction.text import HashingVectorizer

vect = HashingVectorizer(decode_error='ignore', n_features=2**21,
                preprocessor=None, tokenizer=tokenizer)