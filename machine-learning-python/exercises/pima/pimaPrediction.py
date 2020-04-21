import pandas as pd                                     # pandas is a dataframe library
import matplotlib.pyplot as plt                         # matplotlib.pyplot plots movie-recommendations
from sklearn.model_selection import train_test_split    # cross_validation provides splitting the movie-recommendations
from sklearn import metrics, impute                     # sklearn.preprocessing imputes the movie-recommendations
from sklearn.naive_bayes import GaussianNB              # import naive base algorithm
from sklearn.ensemble import RandomForestClassifier     # import random forest algorithm
from sklearn.linear_model import LogisticRegression     # import logistic regression algorithm
from sklearn.linear_model import LogisticRegressionCV   # import logistic regression cross validation algorithm
from sklearn import metrics                             # import metrics to preform performance checks

desired_width = 320
desired_columns = 10
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', desired_columns)

# do ploting inline instead of in a separate window
# %matplotlib inline


# Helper function that displays correlation by color. Red is most correlated, Blue least.
def plot_corr(df, size=11):
    corr = df.corr()  # movie-recommendations frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)  # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks
    plt.show()


# Definition of features:
#
# | Feature      | Description                                               | Comments
# |--------------|-----------------------------------------------------------|
# | num_preg     | number of pregnancies                                     |
# | glucose_conc | concentration a 2 hours in an oral glucose tolerance test |
# | diastolic_bp | Diastolic blood pressure (mm Hg)                          |
# | thickness    | Triceps skin fold thickness (mm)                          |
# |insulin       | 2-Hour serum insulin (mu U/ml)                            |
# | bmi          |  Body mass index (weight in kg/(height in m)^2)           |
# | diab_pred    |  Diabetes pedigree function                               |
# | Age (years)  | Age (years)                                               |
# | skin         | ????                                                      | What is this?
# | diabetes     | Class variable (1=True, 0=False)                          | Why is our movie-recommendations boolean (True/False)?

df = pd.read_csv("./data/pima-data.csv")

# (rows, columns) => presents the numbers of rows and the number of columns
print("Shape:")
print(df.shape)
print("")

# presents the first 5 rows
print(df.head(5))
print("")

# presents the last 5 rows
print(df.tail(5))
print("")

# check for null values
print("Check for null values:")
print(df.isnull().values.any())
print("")

# check for correlations
# plot_corr(df)

print(df.corr())
print("")

# the skin and thickness columns are correlated 1 to 1. Dropping the skin column
del df['skin']
print(df.head())
print("")

# check for additional correlations
# plot_corr(df)

# check movie-recommendations types
print(df.head(5))
print("")
diabetes_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)
print(df.head(5))
print("")

# check true/false ratio
num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])
print("Number of True cases: {0} ({1:2.2f}%)".format(num_true, (num_true / (num_true + num_false)) * 100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, (num_false / (num_true + num_false)) * 100))
print("")

# splitting the movie-recommendations, 70% fro training and 30% for testing
features_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

X = df[features_col_names].values  # predictor feature columns (8 X m)
y = df[predicted_class_names].values  # predictor class (1=true, 0=false) column (1 X m)
split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)

# we need to insure we have the desired 70% train, 30% test split of the movie-recommendations
print("{0:0.2f}% in training set".format((len(X_train) / len(df.index)) * 100))
print("{0:0.2f}% in test set".format((len(X_test) / len(df.index)) * 100))
print("")

# verifying predicted value was split correctly
print("Original True  : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 1]), (len(df.loc[df['diabetes'] == 1])/len(df.index)) * 100.0))
print("Original False : {0} ({1:0.2f}%)".format(len(df.loc[df['diabetes'] == 0]), (len(df.loc[df['diabetes'] == 0])/len(df.index)) * 100.0))
print("")
print("Training True  : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train) * 100.0)))
print("Training False : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train) * 100.0)))
print("")
print("Test True      : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test) * 100.0)))
print("Test False     : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test) * 100.0)))
print("")

# hidden missing values
print("# rows in dataframe {0}".format(len(df)))
print("# rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc'] == 0])))
print("# rows missing diastolic_bp: {0}".format(len(df.loc[df['diastolic_bp'] == 0])))
print("# rows missing thickness: {0}".format(len(df.loc[df['thickness'] == 0])))
print("# rows missing insulin: {0}".format(len(df.loc[df['insulin'] == 0])))
print("# rows missing bmi: {0}".format(len(df.loc[df['bmi'] == 0])))
print("# rows missing diab_pred: {0}".format(len(df.loc[df['diab_pred'] == 0])))
print("# rows missing age: {0}".format(len(df.loc[df['age'] == 0])))

# impute with mean all 0 readings
fill_0 = impute.SimpleImputer(missing_values=0, strategy="mean")
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

# training initial algorithm - Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train.ravel())
print("")

# performance on training movie-recommendations
nb_predict_train = nb_model.predict(X_train)
print("Naive Base - Train Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, nb_predict_train)))
print("")

# predict values using the testing movie-recommendations
nb_predict_test = nb_model.predict(X_test)
print("Naive Base - Test Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_test)))
print("")

# metrics
# True Negative  | False Positive
# False Negative | True Positive
print("Confusion Matrix")
print("{0}".format(metrics.confusion_matrix(y_test, nb_predict_test)))
print("")

# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
print("Classification Report")
print(metrics.classification_report(y_test, nb_predict_test))
print("")

# training initial algorithm - Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train.ravel())
rf_predict_train = rf_model.predict(X_train)
print("Random Forest - Train Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))
rf_predict_test = rf_model.predict(X_test)
print("Random Forest - Test Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))
print("")
print(metrics.confusion_matrix(y_test, rf_predict_test) )
print("")
print("Classification Report")
print(metrics.classification_report(y_test, rf_predict_test))

# training initial algorithm - Logistic Regression
lr_model = LogisticRegression(C=0.7, random_state=42)
lr_model.fit(X_train, y_train.ravel())
rf_predict_train = lr_model.predict(X_train)
print("Logistic Regression - Train Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train, rf_predict_train)))
rf_predict_test = lr_model.predict(X_test)
print("Logistic Regression - Test Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, rf_predict_test)))
print("")
print(metrics.confusion_matrix(y_test, rf_predict_test) )
print("")
print("Classification Report")
print(metrics.classification_report(y_test, rf_predict_test))

# setting regularization parameter
C_start = 0.1
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, random_state=42, solver='liblinear')
    lr_model_loop.fit(X_train, y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if (recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test

    C_val = C_val + C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))

# % matplotlib inline
plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")
# plt.show()

# logistic regression with class_weight='balanced
C_end = 5
C_inc = 0.1

C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, class_weight="balanced", random_state=42, solver='liblinear', max_iter=10000)
    lr_model_loop.fit(X_train, y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test)
    recall_scores.append(recall_score)
    if (recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test

    C_val = C_val + C_inc

best_score_C_val = C_values[recall_scores.index(best_recall_score)]
print("1st max value of {0:.3f} occured at C={1:.3f}".format(best_recall_score, best_score_C_val))

# % matplotlib inline
plt.plot(C_values, recall_scores, "-")
plt.xlabel("C value")
plt.ylabel("recall score")
# plt.show()

# --------------

# After we achieved our goal, we define the best score C value in our model
lr_model = LogisticRegression(class_weight="balanced", C=best_score_C_val, random_state=42, solver='liblinear')
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

# training metrics
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))
print(metrics.confusion_matrix(y_test, lr_predict_test))
print("")
print("Classification Report")
print(metrics.classification_report(y_test, lr_predict_test))
print("Best Logistic Regression - Test Accuracy: {0:.4f}".format(metrics.recall_score(y_test, lr_predict_test)))
print("")

# --------------

# Logistic Regression cross validation
lr_cv_model = LogisticRegressionCV(n_jobs=-1, random_state=42, Cs=3, cv=10, refit=False, class_weight="balanced", solver='liblinear')
lr_cv_model.fit(X_train, y_train.ravel())
lr_cv_predict_test = lr_cv_model.predict(X_test)
# training metrics
print("Logistic Regression cross valdation - Test Accuracy: {0:.4f}".format(metrics.recall_score(y_test, lr_cv_predict_test)))
print(metrics.confusion_matrix(y_test, lr_cv_predict_test))
print("")
print("Classification Report")
print(metrics.classification_report(y_test, lr_cv_predict_test))
