from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


def result_string(x):
    return {
        0: "Negative",
        1: "Positive",
    }[x]


with open("./movie-recommendations/imdb_labelled.txt", "r", encoding="utf8") as text_file:
    lines = text_file.read().split('\n')
with open("./movie-recommendations/amazon_cells_labelled.txt", "r", encoding="utf8") as text_file:
    lines += text_file.read().split('\n')
with open("./movie-recommendations/yelp_labelled.txt", "r", encoding="utf8") as text_file:
    lines += text_file.read().split('\n')

lines = [line.split('\t') for line in lines if len(line.split('\t')) == 2 and line.split('\t') != '']

train_documents = [line[0] for line in lines]
train_labels = [int(line[1]) for line in lines]

count_vectorizer = CountVectorizer(binary='True')
train_documents = count_vectorizer.fit_transform(train_documents)

# training the movie-recommendations
classifier = BernoulliNB().fit(train_documents, train_labels)

# predict inputs
sen_input = "this is the best movie"
result = classifier.predict(count_vectorizer.transform([sen_input]))
print("input: {0}, prediction: {1}".format(sen_input, result_string(result[0])))

sen_input = "this is the worst movie"
result = classifier.predict(count_vectorizer.transform([sen_input]))
print("input: {0}, prediction: {1}".format(sen_input, result_string(result[0])))
