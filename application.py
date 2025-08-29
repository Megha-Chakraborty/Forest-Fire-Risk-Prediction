##USING FLASK

from flask import Flask, request, render_template
import numpy as np
import pickle

application = Flask(__name__)
app = application

# import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template('home.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
            result = ridge_model.predict(new_data_scaled)

            # Pass the result and the form data back to the template
            return render_template('home.html', results=result[0], form_data=request.form)
        except Exception as e:
            # Pass the error and form data back to the template
            return render_template('home.html', error=f"Error: {e}", form_data=request.form)
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)