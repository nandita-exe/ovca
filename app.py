from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
app = Flask(__name__)

# Load your trained ML model
model = joblib.load('model5.pkl')  # Replace 'your_model.pkl' with your model file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/faqs')
def faqs():
    return render_template('faq.html')


@app.route('/predict', methods=['POST'])
def predict():
    # # Get data from the request
    # data = request.get_json()

    # values = [float(val) for val in data.values()]

    features = [
float(request.form['Age']),
        float(request.form['ALB']),
        float(request.form['ALP']),
        float(request.form['AST']),
        float(request.form['CA125']),
        float(request.form['HE4']),
        float(request.form['HGB']),
        float(request.form['IBIL']),
        float(request.form['LYM#']),
        float(request.form['LYM%']),
        int(request.form['Menopause']),
        float(request.form['NEU']),
        float(request.form['PCT']),
        float(request.form['PLT']),
        float(request.form['TBIL'])
    ]
    # Convert the list of values into a NumPy array
    input_array = np.array(features)

    # Reshape the array to ensure it's a 2D array (if needed)
    input_array = input_array.reshape(1, -1)
    prediction = model.predict(input_array)  # Replace 'data' with your actual input data
    return render_template('prediction.html', prediction=prediction[0])

    # Return the prediction as JSON response
    # return jsonify({'prediction': prediction.tolist()})  # Convert prediction to list if needed

if __name__ == '__main__':
    app.run(debug=True)
