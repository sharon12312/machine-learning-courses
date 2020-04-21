import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

diabetes_data = pd.read_csv('data/diabetes_processed.csv')
print(diabetes_data.head())

X = diabetes_data.drop('Outcome', axis=1)
Y = diabetes_data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# initial our models
log_clf = LogisticRegression(C=1, solver='liblinear')
svc_clf = SVC(C=1, kernel='linear', gamma='auto')
naive_clf = GaussianNB()

voting_clf_hard = VotingClassifier(estimators=[('lr', log_clf), ('svc',svc_clf), ('naive', naive_clf)], voting='hard')
voting_clf_hard.fit(x_train, y_train)

y_pred = voting_clf_hard.predict(x_test)
print(accuracy_score(y_test, y_pred))

for clf_hard in (log_clf, svc_clf, naive_clf, voting_clf_hard):
    clf_hard.fit(x_train, y_train)
    y_pred = clf_hard.predict(x_test)
    print(clf_hard.__class__.__name__, accuracy_score(y_test, y_pred))

svc_clf_soft = SVC(C=1, kernel='linear', gamma='auto', probability=True)

voting_clf_soft = VotingClassifier(estimators=[('lr', log_clf), ('svc',svc_clf_soft), ('naive', naive_clf)], voting='soft', weights=[0.25, 0.5, 0.25])

for clf_soft in (log_clf, svc_clf_soft, naive_clf, voting_clf_hard):
    clf_soft.fit(x_train, y_train)
    y_pred = clf_soft.predict(x_test)
    print(clf_soft.__class__.__name__, accuracy_score(y_test, y_pred))

