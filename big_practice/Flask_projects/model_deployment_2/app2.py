
import pandas as pd
import pickle
import math
import numpy as np
from flask import Flask,  request, jsonify
from sklearn.metrics import roc_curve

#create an instance of Flask
app = Flask(__name__)

with open('/Users/aitor/Desktop/Máster Ciencia de Datos/Aprendizaje automático/Machine_Learning/big_practice/Models/RF_PCA.pickle', 'rb') as f:
    model=pickle.load(f)

@app.route('/prediction',methods=["POST"])#POST nos sirve para introducir datos en el html

def predict():
    json_ = request.json
    data_dictionary=dict()
    for variable in json_:
        if variable == "C_MNTH":
            month = int(json_[variable])
            NA_cos_C_MNTH = np.cos(2*math.pi*month/12)
            NA_sin_C_MNTH = np.sin(2*math.pi*month/12)
            data_dictionary["NA_cos_C_MNTH"] = NA_cos_C_MNTH
            data_dictionary["NA_sin_C_MNTH"] = NA_sin_C_MNTH
        elif variable == "C_HOUR":
            hour = int(json_[variable])
            NA_cos_C_HOUR = np.cos(2*math.pi*hour/24)
            NA_sin_C_HOUR = np.sin(2*math.pi*hour/24)
            data_dictionary["NA_cos_C_HOUR"] = NA_cos_C_HOUR
            data_dictionary["NA_sin_C_HOUR"] = NA_sin_C_HOUR
        elif variable == "C_WDAY":
            wday = int(json_[variable])
            NA_cos_C_WDAY = np.cos(2*math.pi*wday/7)
            NA_sin_C_WDAY = np.sin(2*math.pi*wday/7)
            data_dictionary["NA_cos_C_WDAY"] = NA_cos_C_WDAY
            data_dictionary["NA_sin_C_WDAY"] = NA_sin_C_WDAY
        elif variable == "num_C_YEAR":
            value_c_year = int(json_[variable])
            mean_c_year_test = 2006.0155218348198
            sdv_c_year_test = 4.567695934222996
            new_value_year = (value_c_year - mean_c_year_test)/sdv_c_year_test
            data_dictionary["num_C_YEAR"] = new_value_year
        elif variable == "num_Random":
            value_Random = int(json_[variable])
            mean_Random_test = 49.51538088565281
            sdv_Random_test = 28.850311918400497
            new_value_Random = (value_Random-mean_Random_test)/sdv_Random_test
            data_dictionary["num_Random"] = new_value_Random
        elif variable == "num_C_VEHS":
            value_c_vehs = int(json_[variable])
            mean_c_vehs_test = 2.0971652406757
            sdv_c_vehs_test = 1.3324953572064602
            new_value_vehs = (value_c_vehs-mean_c_vehs_test)/sdv_c_vehs_test
            data_dictionary["num_C_VEHS"] = new_value_vehs
        else:
            data_dictionary[variable] = json_[variable]
    query_df = pd.DataFrame(data_dictionary, index = [0])
    prediction = model.predict(query_df)
    prediction_proba = model.predict_proba(query_df)
    yhat = prediction_proba[:, 1]
    fpr, tpr, thresholds = roc_curve(ytest, yhat)
    gmeans = np.sqrt(tpr* (1-fpr))
    ix = np.argmax(np.sqrt(tpr * (1-fpr)))
    predictions_new=(prediction_proba[:,1] >= thresholds[ix]).astype(int)
    return jsonify({"prediction":int(predictions_new[0])})

if __name__ == "__main__":
    app.run(debug=True)

