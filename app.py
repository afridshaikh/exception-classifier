from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

models = ['NaiveBayes', 'SVM', 'XGBoost']
vectorizer_file = 'model/tfidf_vectorizer.pkl'
tfidf_vectorizer = joblib.load(vectorizer_file)

def load_model(name):
    model_path = f'model/{name}.pkl'
    return joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input_data']
    inputs_tfidf = tfidf_vectorizer.transform(data)
    result = {}
    
    for model_name in models:
        model = load_model(model_name)
        predictions = model.predict(inputs_tfidf)
        result[model_name] = predictions.tolist()
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
