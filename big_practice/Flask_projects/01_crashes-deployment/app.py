
import pandas as pd
import pickle
import math
import numpy as np
from flask import Flask,  request, jsonify

#create an instance of Flask
app = Flask(__name__)

model=pickle.load(open('Models/RF.pkl', 'rb'))

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
    predictions_proba = model.predict_proba(query_df)
    prediction=(predictions_proba[:,1] >= 0.9735).astype(int)
    if prediction == 0:
        a = "this is, there is at leats one fatality"
    else:
        a = "this is, there are no fatalitites"
    
    return jsonify({"prediction":int(prediction)}, a)

if __name__ == "__main__":
    app.run(debug=True)

