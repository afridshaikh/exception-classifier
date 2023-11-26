import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

df = pd.read_csv('data/exception_training_data_large.csv')
X_train, X_test, y_train, y_test = train_test_split(df['Exception'], df['Category'], test_size=0.2, random_state=42)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
vectorizer = 'model/tfidf_vectorizer.pkl'
joblib.dump(tfidf_vectorizer, vectorizer)

classifiers = {
    'NaiveBayes': MultinomialNB(),
    'SVM': SVC(kernel='linear'),
    'GradientBoosting': GradientBoostingClassifier()
}

def evaluate_classifier(y_true, y_pred, classifier_name):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print(f"{classifier_name} Accuracy: {accuracy}")
    print(f"{classifier_name} Classification Report:\n{report}")


def save_model(model, name):
    Path("model").mkdir(parents=True, exist_ok=True)
    model_path = f'model/{name}.pkl'
    joblib.dump(model, model_path)


for name, classifier in classifiers.items():
    classifier.fit(X_train_tfidf, y_train)
    predictions = classifier.predict(X_test_tfidf)
    evaluate_classifier(y_test, predictions, name)
    print("="*40)
    save_model(classifier, name)

