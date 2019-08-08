import pandas as pd
import numpy as np
import json
import re
from flask import Flask
from flask import request
from flask import jsonify
from flask_mysqldb import MySQL
from sklearn.externals import joblib


app = Flask(__name__)

# SQL config
app.config['MYSQL_HOST'] = 'remotemysql.com'
app.config['MYSQL_USER'] = 'vMg935rEqf'
app.config['MYSQL_PASSWORD'] = 'SkgUJ4EGYq'
app.config['MYSQL_DB'] = 'vMg935rEqf'

mysql = MySQL(app)

model = joblib.load('model.pkl')

f = open('./features.txt', 'r')
features = f.read().split(',')
f = open('./symptoms.txt', 'r')
symptoms = f.read().split(',')
f.close()

feature_dict = {}
for i, f in enumerate(features):
    feature_dict[f] = i


@app.route('/predict', methods=['POST'])
def predict():
    search = []
    data = request.get_json()['symptoms']

    cur = mysql.connection.cursor()

    for x in data:
        cur.execute(
            '''SELECT DISTINCT Symptom_CUI FROM vMg935rEqf.`disease-symptom` WHERE Symptom='{0}';'''.format(x))
        search.append(cur.fetchone()[0])

    sample = np.zeros((len(features),), dtype=np.int)
    sample = sample.tolist()
    for i, s in enumerate(search):
        sample[feature_dict[s]] = 1

    sample = np.array(sample).reshape(1, len(sample))

    results = model.predict_proba(sample)[0]

    diseases = []

    for x in model.classes_:
        cur.execute(
            '''SELECT DISTINCT Disease FROM vMg935rEqf.`disease-symptom` WHERE Disease_CUI = '{0}';'''.format(x))
        diseases.append(cur.fetchone()[0])

    prob_per_class_dictionary = list(zip(diseases, results))

    results_ordered_by_probability = list(map(
        lambda x: {"disease": x[0], "prop": x[1] * 100},
        sorted(zip(diseases, results), key=lambda x: x[1], reverse=True)))

    return jsonify(results_ordered_by_probability[0:10])


@app.route('/symptom', methods=['GET'])
def symptom():
    return jsonify(symptoms)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
