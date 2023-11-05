import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('data/exception_training_data_large.csv')
X_train, X_test, y_train, y_test = train_test_split(df['Exception'], df['Category'], test_size=0.2, random_state=42)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 1. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)
y_pred_rf = rf_classifier.predict(X_test_tfidf)

# 2. Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_tfidf, y_train)
y_pred_dt = dt_classifier.predict(X_test_tfidf)

# 3. Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train_tfidf, y_train)
y_pred_gb = gb_classifier.predict(X_test_tfidf)

# Evaluate the classifiers
def evaluate_classifier(y_true, y_pred, classifier_name):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    print(f"{classifier_name} Accuracy: {accuracy}")
    print(f"{classifier_name} Classification Report:\n{report}")

evaluate_classifier(y_test, y_pred_rf, "Random Forest")
evaluate_classifier(y_test, y_pred_dt, "Decision Tree")
evaluate_classifier(y_test, y_pred_gb, "Gradient Boosting")

