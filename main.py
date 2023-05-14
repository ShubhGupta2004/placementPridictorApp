from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)



def create_app():
    app = Flask(__name__)
    return app


@app.route('/')
def index():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    profile_score = request.form.get('profile_score')
    print(cgpa)

    input_query = np.array([[cgpa, iq, profile_score]])

    result = model.predict(input_query)[0]

    return jsonify({'placement': str(result), 'input': cgpa, 'iq': iq, 'ps': profile_score})


if __name__ == '__main__':
    app.run(debug=True)
