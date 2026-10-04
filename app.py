from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("banknote_auth_model.joblib")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        variance = float(request.form['variance'])
        skewness = float(request.form['skewness'])
        curtosis = float(request.form['curtosis'])
        entropy = float(request.form['entropy'])

        variance = round(variance, 5)
        skewness = round(skewness, 5)
        curtosis = round(curtosis, 5)
        entropy = round(entropy, 5)

        features = np.array([[variance, skewness, curtosis, entropy]])
        prediction = model.predict(features)[0]

        result = "Genuine " if prediction == 1 else "Fake "

        return render_template("index.html", prediction_text=f"Bank Note is: {result}")

    except:
        return render_template("index.html", prediction_text="Please enter valid numbers")

if __name__ == "__main__":
    app.run(debug=True)
