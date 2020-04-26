import pandas as pd
from sklearn import preprocessing

exam_data = pd.read_csv('./data/exams.csv', quotechar='"')
print(exam_data.head())

math_average = exam_data['math score'].mean()
reading_average = exam_data['reading score'].mean()
writing_average = exam_data['writing score'].mean()

print('Math Avg: ', math_average)
print('Reading Avg: ', reading_average)
print('Writing Avg: ', writing_average)

# standardize the data's scores
# all the score will be with the same range
exam_data[['math score']] = preprocessing.scale(exam_data[['math score']])
exam_data[['reading score']] = preprocessing.scale(exam_data[['reading score']])
exam_data[['writing score']] = preprocessing.scale(exam_data[['writing score']])

print(exam_data.head())

math_average = exam_data['math score'].mean()
reading_average = exam_data['reading score'].mean()
writing_average = exam_data['writing score'].mean()

print('Math Avg: ', math_average)
print('Reading Avg: ', reading_average)
print('Writing Avg: ', writing_average)

# --------------------

le = preprocessing.LabelEncoder()

# convert from male/female to 0/1 by using fit_transform() function
exam_data['gender'] = le.fit_transform(exam_data['gender'].astype(str))
print(exam_data.head())

# presents the male/female classes
print(le.classes_)

# one-hot encoding using get_dummies() function
print(exam_data['race/ethnicity'].unique())
print(pd.get_dummies(exam_data['race/ethnicity']))
exam_data = pd.get_dummies(exam_data, columns=['race/ethnicity'])
exam_data = pd.get_dummies(exam_data, columns=['parental level of education', 'lunch', 'machine-learning-my-models preparation course'])
print(exam_data.columns)
print(exam_data)
