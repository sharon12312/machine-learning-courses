import pandas as pd
import matplotlib.pyplot as plt
import re, string, unicodedata
import nltk
import inflect
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('wordnet')
# nltk.download('punkt')

data = pd.read_csv('../data/reviews_Baby_5_final_dataset.csv')
data = data[['reviewText']]
data = data[:6]

# ------------ Tokenization ------------ #

documents = list(data['reviewText'])
document_word_tokens = []

for document in documents:
    tokenize_word = word_tokenize(document)
    document_word_tokens.append(tokenize_word)

print(documents[0])
print(document_word_tokens[0])

fdist = FreqDist(document_word_tokens[0])
print(fdist.most_common(15))  # most common 15 words in this document

# visualization
plt.figure(figsize=(18, 6))
# fdist.plot(cumulative=False)
# plt.show()

# ------------ Normalization ------------ #

def to_lowercase(documents):
    documents_list = []
    for document in documents:
        new_word = document.lower()
        documents_list.append(new_word)
    return documents_list


def remove_punctuation(documents):
    documents_list = []
    for document in documents:
        new_word = re.sub('[^\w\s]', '', document)
        if new_word != '':
            documents_list.append(new_word)
    return documents_list


def replace_numbers(documents):
    documents_list = []
    inf_engine = inflect.engine()

    for document in documents:
        final_word_list = []
        words = document.split()

        for word in words:
            if word.isdigit():
                final_word_list.append(inf_engine.number_to_words(word))
            else:
                final_word_list.append(word)

        documents_list.append(" ".join(final_word_list))

    return documents_list


def lemmatize_verbs(documents):
    lemmatizer = WordNetLemmatizer()
    documents_list = []

    for document in documents:
        final_word_list = []
        words = document.split()

        for word in words:
            final_word_list.append(lemmatizer.lemmatize(word, pos='v'))
        documents_list.append(" ".join(final_word_list))

    return documents_list


def normalization(documents):
    documents = to_lowercase(documents)
    documents = remove_punctuation(documents)
    documents = replace_numbers(documents)
    documents = lemmatize_verbs(documents)
    return documents


print(documents[2])
print(normalization(documents)[2])

# ------------ Vectorization ------------ #

count_vectorizer = CountVectorizer()
count_vectorizer.fit(documents)

# presents the frequency of our vocabulary
print(count_vectorizer.vocabulary_)
print(len(count_vectorizer.vocabulary_))

doc_terms = count_vectorizer.fit_transform(documents)
print(doc_terms.shape)  # (number of documents, numbers of vocabulary words frequency)

print('(Doc, WordIndex.): NumOccurences')
print(doc_terms)
print(doc_terms[0].toarray())  # presents an array contains the frequency in eac index
print(count_vectorizer.get_feature_names()[336])  # presents the specific word in index 336

stop_words_df = pd.read_csv('../data/stopwords_english.csv', engine='python', header=None, usecols=[0], names=['words'])
print(stop_words_df.sample(5))  # words which contains no information

# first approach
stop_words = set(stop_words_df.words.unique())
count_vectorizer = CountVectorizer(stop_words=stop_words)
count_vectorizer.fit(documents)

print(len(count_vectorizer.vocabulary_))  # after removing stop words

# second approach
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)  # set tfidf score for each word
tfidf_vectorizer.fit(documents)
print(tfidf_vectorizer.vocabulary_)

df = pd.DataFrame(tfidf_vectorizer.idf_, index=tfidf_vectorizer.get_feature_names(), columns=['IDF Score'])
print(df.sort_values(by=['IDF Score']).head(10))  # most common words
print(df.sort_values(by=['IDF Score']).tail(10))  # most uncommon words

tfidf_vectors = tfidf_vectorizer.fit_transform(documents)
print(tfidf_vectors[0].todense())  # present the document's (in index 0) vocabulary vector
print(tfidf_vectorizer.get_feature_names()[28])  # get the word in index 28
print(tfidf_vectors[:, 28].todense())  # the vector for word in index 28
