import joblib

models = ['NaiveBayes', 'SVM', 'GradientBoosting']

vectorizer_file = 'model/tfidf_vectorizer.pkl'
tfidf_vectorizer = joblib.load(vectorizer_file)

custom_input = ["NullPointerException: Null value found", "ExpiredOauthToken: The oauth token has expired" ]
custom_input_tfidf = tfidf_vectorizer.transform(custom_input)

def predict(name, inputs):
    model_path = f'model/{name}.pkl'
    model = joblib.load(model_path)
    predictions = model.predict(inputs)
    for input, prediction in zip(custom_input, predictions):
        print(f"Input: {input}")
        print(f"Predicted Category by {name}: {prediction}")
        print("\n")
    print("\n" + "------"*10 + "\n")
 
for model in models:
    predict(model, custom_input_tfidf)