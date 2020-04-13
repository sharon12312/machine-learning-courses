from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

with open("./data/imdb_labelled.txt", "r", encoding="utf8") as text_file:
    lines = text_file.read().split('\n')

lines = [line.split('\t') for line in lines if len(line.split('\t')) == 2 and line.split('\t') != '']

train_documents = [line[0] for line in lines]

tfidf_vectorizer = TfidfVectorizer(max_df=5, min_df=2, stop_words="english")
train_documents = tfidf_vectorizer.fit_transform(train_documents)

km = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1, verbose=True)
km.fit(train_documents)
print("--------------")

count = 0
label = 2
print("Lines of label {} - ".format(label))
for i in range(len(lines)):
    if count > 3:
        break
    if km.labels_[i] == label:
        print("Line {0}: {1} ".format(i, lines[i][0]))
        count += 1