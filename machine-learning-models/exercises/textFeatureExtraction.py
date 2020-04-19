import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

corpus = ['This is the first document.',
          'This is the second document.',
          'Third document. Document number three',
          'Number four. To repeat, number four']

# --- Count Vectorizer ---

vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(corpus)
print(vectorizer.vocabulary_)  # all the words ids in our vocabulary
print(bag_of_words)  # (number of a document, integer unique id for a specific word) | frequency
print(vectorizer.vocabulary_.get('document'))  # get the unique id of 'document' word

# presents the words as columns and their frequency in each document index from 0 to N
print(pd.DataFrame(bag_of_words.toarray(), columns=vectorizer.get_feature_names()))

# --- TfIdf Vectorizer ---
vectorizer = TfidfVectorizer()
bag_of_words = vectorizer.fit_transform(corpus)
print(vectorizer.vocabulary_)  # all the words ids in our vocabulary
print(bag_of_words)  # (number of a document, integer unique id for a specific word) | score
print(vectorizer.vocabulary_.get('document'))  # get the unique id of 'document' word

# presents the words as columns and their frequency in each document index from 0 to N
print(pd.DataFrame(bag_of_words.toarray(), columns=vectorizer.get_feature_names()))

# --- Hashing Vectorizer ---
vectorizer = HashingVectorizer(n_features=8)  # n_features => the size of our feature vector (number of buckets)
feature_vector = vectorizer.fit_transform(corpus)
print(feature_vector)  # (number of a document, bucket (max: n_features) | normalized frequency number
