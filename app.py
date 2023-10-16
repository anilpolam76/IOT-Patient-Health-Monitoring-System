from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('capstone_project.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
 return render_template('index.html')
standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
 
 if request.method == 'POST':
 age = int(request.form['age'])
 sex = int(request.form['sex'])
 chest_pain_type = int(request.form['chest_pain_type'])
 resting_bp_s = int(request.form['resting_bp_s'])
 cholesterol = int(request.form['cholesterol'])
 fasting_blood_sugar = int(request.form['fasting_blood_sugar'])
 resting_ecg = int(request.form['resting_ecg'])
 max_heart_rate = int(request.form['max_heart_rate'])
 exercise_angina = int(request.form['exercise_angina'])
 oldpeak = float(request.form['oldpeak'])
 ST_slope = int(request.form['ST_slope'])
 
prediction=model.predict([[age,sex,chest_pain_type,resting_bp_s,cholesterol,fasting_blood_sugar,resting
_ecg,max_heart_rate,exercise_angina,oldpeak,ST_slope]])
 output=round(prediction[0],2)
 
 
 if output<0:
return render_template('index.html',prediction_texts="Hurray..! You are healthy")
 elif output>0:
 return render_template('index.html',prediction_texts="You may have a heart disease please consult 
doctor")
 else:
 return render_template('index.html',prediction_text="Hurray..! You are healthy")
 else:
 return render_template('index.html')
if __name__=="__main__":
 app.run(debug=True)