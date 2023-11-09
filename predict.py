import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model
rf_classifier_model = 'model/rf_classifier_model.pkl'
dt_classifier_model = 'model/dt_classifier_model.pkl'
gb_classifier_model = 'model/gb_classifier_model.pkl'
vectorizer_file = 'model/tfidf_vectorizer.pkl'
tfidf_vectorizer = joblib.load(vectorizer_file)
rf_classifier = joblib.load(rf_classifier_model)
dt_classifier = joblib.load(dt_classifier_model)
gb_classifier = joblib.load(gb_classifier_model)

# Prepare a custom input for prediction
custom_input = ["java.lang.NullPointerException: Null value found", "java.lmn.SomeRandom: Null value found", ]

# Vectorize the custom input
custom_input_tfidf = tfidf_vectorizer.transform(custom_input)

# Make predictions
rf_classifier_model_predictions = rf_classifier.predict(custom_input_tfidf)
dt_classifier_model_predictions = dt_classifier.predict(custom_input_tfidf)
gb_classifier_model_predictions = gb_classifier.predict(custom_input_tfidf)

# Print the predictions
for input, prediction in zip(custom_input, rf_classifier_model_predictions):
    print(f"Input: {input}")
    print(f"Predicted Category by Random Forest: {prediction}")

print("\n" + "------"*10 + "\n")

for input, prediction in zip(custom_input, dt_classifier_model_predictions):
    print(f"Input: {input}")
    print(f"Predicted Category by Decision Tree: {prediction}")

print("\n" + "------"*10 + "\n")

for input, prediction in zip(custom_input, gb_classifier_model_predictions):
    print(f"Input: {input}")
    print(f"Predicted Category by Gradient Boosting: {prediction}")    

