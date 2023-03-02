import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from create_processed_dataset import data_path

year = 2018
race = 'house'
df = pd.read_csv(data_path + str(year) + '/' + race + '_bow_emb_processed.csv')

max_features = 500
text_bow_rows = ['text_bow_'+ str(i) for i in range(max_features)]

X = df[text_bow_rows].values
y = (df['vote_diff'] > 0).astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print("LogisticRegression Accuracy %.3f" % metrics.accuracy_score(y_test, y_pred))