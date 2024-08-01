import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

data = pd.read_csv(r"C:\Users\ANAND\Downloads\FILE\SMS_SPAM_DETECTION\spam.csv" , encoding='latin-1')

data.drop_duplicates(inplace=True)
data['label'] = data['v1'].map({'ham': 'ham', 'spam': 'spam'})
x = data['v2']
y = data['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()

x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)

classifier = MultinomialNB()

classifier.fit(x_train_tfidf, y_train)

x_test_tfidf = tfidf_vectorizer.transform(x_test)

y_pred = classifier.predict(x_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(y_test, y_pred, target_names=['Legitimate SMS', 'Spam SMS'])

progress_bar = tqdm(total=100, position=0, leave=True)

for i in range(10, 101, 10):
    progress_bar.update(10)
    progress_bar.set_description(f'Progress: {i}%')

progress_bar.close()

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
