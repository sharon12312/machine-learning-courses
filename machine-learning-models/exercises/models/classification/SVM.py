from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='machine-learning-my-models', shuffle=True)

print(twenty_train.keys())
print(twenty_train.data[0])
print(twenty_train.target_names)
print(twenty_train.target)

# first approach
count_vector = CountVectorizer()
X_train_counts = count_vector.fit_transform(twenty_train.data)
print(X_train_counts.shape)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)
print(X_train_tfidf[0])

# check penalty for 'l1' as well to see more results
clf_svc = LinearSVC(penalty='l2', dual=False, tol=1e-3)
clf_svc.fit(X_train_tfidf, twenty_train.target)

# second approach
# instead of writing all the steps, we can perform them using pipeline
clf_svc_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC(penalty='l2', dual=False, tol=0.001))
])

clf_svc_pipeline.fit(twenty_train.data, twenty_train.target)

predicted = clf_svc_pipeline.predict(twenty_test.data)
acc_svm = accuracy_score(twenty_test.target, predicted)
print(acc_svm)

# third approach (less good model)
# with Count Vectorizer only
clf_svc_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LinearSVC(penalty='l2', dual=False, tol=0.001))
])

clf_svc_pipeline.fit(twenty_train.data, twenty_train.target)

predicted = clf_svc_pipeline.predict(twenty_test.data)
acc_svm = accuracy_score(twenty_test.target, predicted)
print(acc_svm)
