from flask import Flask, render_template, request
import pandas as pd
from pycaret.regression import load_model, predict_model
import pickle

app = Flask(__name__)

# Load the saved PyCaret model
loaded_final_model = pickle.load(open("/Users/jm/Desktop/PSG/team3_blend5_try2.pkl","rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from HTML form
        time_under90 = float(request.form['time_under90'])
        snoring_size = float(request.form['snoring_size'])
        spo2_avg = float(request.form['spo2_avg'])
        spo2_min = float(request.form['spo2_min'])
        bmi = float(request.form['bmi'])

        # Create a DataFrame from input data
        input_data = pd.DataFrame([[time_under90, snoring_size, spo2_avg, spo2_min, bmi]],
                                  columns=['Time_under90(Min)', 'Snoring Size(1:mild,2:morderate,3:severe)',
                                           'SpO2_AVG', 'SpO2_MIN', 'BMI'])

        # Make AHI prediction using the loaded model
        prediction = predict_model(loaded_final_model, data=input_data)
        predicted_ahi = prediction['Label'][0]

        return render_template("home.html", prediction_text=f'Predicted AHI: {predicted_ahi:.2f}')

if __name__ == '__main__':
    app.run(debug=True)