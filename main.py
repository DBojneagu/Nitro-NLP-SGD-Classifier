################  SGD  ################
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv(r"./nitro-language-processing-2/train_data.csv")
train_data['Final Labels']

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['Text'])
y_train = train_data['Final Labels']

clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-200, random_state=42, max_iter=2000, tol=None)
clf.fit(X_train, y_train)

test_data = pd.read_csv("./nitro-language-processing-2/test_data.csv")

X_test = vectorizer.transform(test_data['Text'])

predictions = clf.predict(X_test)

submission = pd.DataFrame({'Id': test_data['Id'], 'Label': predictions})
submission.to_csv('submission.csv', index=False)