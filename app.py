import os
import sys
import numpy as np
import pandas as pd
from flask import request,render_template,Flask
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        CustomData(
            gender = request.form.get('gender'),
        race_ethnicity = request.form.get('race_ethnicity'),
        parental_level_of_education = request.form.get('parental_level_of_education'),
        lunch = request.form.get('lunch'),
        test_preparation_course = request.form.get('test_preparation_course'),
        reading_score = request.form.get('reading_score'),
        writing_score = request.form.get('writing_score')
        )
