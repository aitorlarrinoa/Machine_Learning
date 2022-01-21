import pandas as pd
import pickle
import math
import numpy as np
from flask import Flask, render_template, request

variables = ['num_C_YEAR', 'C_MNTH', 'C_WDAY','C_HOUR', 'num_C_VEHS', 'num_Random', 'C_CONF', 'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN', 'C_TRAF']

#create an instance of Flask
app = Flask(__name__)

# Cargamos el modelo
model=pickle.load(open('Models/RF.pkl', 'rb'))

# cuando el usuario acceda a la url "/", automáticamente se iniciará la función index().
# la función index() se encargará de devolvernos nuestra html principal crashes.html.
# en nuestro caso crashes.html servirá para que el usuario introduzca los datos.
# render_template se encarga de buscar el html crashes en nuestra carpeta de templates y genera un html a partir de dicho archivo.
@app.route('/')# usamos el método por defecto GET (Queremos extraer datos del html)
def home():
    return render_template("index.html")

# ahora creamos otra instancia de @app.route. En este caso, cuando el usuario acceda a "/predict", se iniciará la función
# predict(). Esta función predict() se encargará de tomar el input proporcionado por el usuario y transformarlo de tal manera
# que pueda ser utilizado para predecir con nuestro modelo previamente cargado.
@app.route('/predict/',methods=["GET", "POST"])#POST nos sirve para introducir datos en el html
def modify_data():
    data_dictionary = dict()
    for variable in variables:
        if variable == "C_MNTH":
            month = int(request.form.get(variable))
            NA_cos_C_MNTH = np.cos(2*math.pi*month/12)
            NA_sin_C_MNTH = np.sin(2*math.pi*month/12)
            data_dictionary["NA_cos_C_MNTH"] = NA_cos_C_MNTH
            data_dictionary["NA_sin_C_MNTH"] = NA_sin_C_MNTH
        elif variable == "C_WDAY":
            wday = int(request.form.get(variable))
            NA_cos_C_WDAY = np.cos(2*math.pi*wday/7)
            NA_sin_C_WDAY = np.sin(2*math.pi*wday/7)
            data_dictionary["NA_cos_C_WDAY"] = NA_cos_C_WDAY
            data_dictionary["NA_sin_C_WDAY"] = NA_sin_C_WDAY
        elif variable == "C_HOUR":
            hour = int(request.form.get(variable))
            NA_cos_C_HOUR = np.cos(2*math.pi*hour/24)
            NA_sin_C_HOUR = np.sin(2*math.pi*hour/24)
            data_dictionary["NA_cos_C_HOUR"] = NA_cos_C_HOUR
            data_dictionary["NA_sin_C_HOUR"] = NA_sin_C_HOUR
        elif variable in ["C_CONF", 'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN', 'C_TRAF']:
            if variable== "C_CONF":
                value = int(request.form.get(variable))
                for i in [1, 2, 3, 4, 5, 6]:
                    if value == i:
                        data_dictionary["ohe_"+variable+"_"+str(i)] = 1
                    else:
                        data_dictionary["ohe_"+variable+"_"+str(i)] = 0
            else:
                value = int(request.form.get(variable))
                for i in [1,2, 3, 4]:
                    if value == i:
                        data_dictionary["ohe_"+variable+"_"+str(i)] = 1
                    else:
                        data_dictionary["ohe_"+variable+"_"+str(i)] = 0
        elif variable == "num_C_YEAR":
            value_c_year = int(request.form.get(variable))
            mean_c_year_test = 2006.0155218348198
            sdv_c_year_test = 4.567695934222996
            new_value_year = (value_c_year - mean_c_year_test)/sdv_c_year_test
            data_dictionary["num_C_YEAR"] = new_value_year
        elif variable == "num_C_VEHS":
            value_c_vehs = int(request.form.get(variable))
            mean_c_vehs_test = 2.0971652406757
            sdv_c_vehs_test = 1.3324953572064602
            new_value_vehs = (value_c_vehs-mean_c_vehs_test)/sdv_c_vehs_test
            data_dictionary["num_C_VEHS"] = new_value_vehs
        else:
            value_Random = int(request.form.get(variable))
            mean_Random_test = 49.51538088565281
            sdv_Random_test = 28.850311918400497
            new_value_Random = (value_Random-mean_Random_test)/sdv_Random_test
            data_dictionary["num_Random"] = new_value_Random

    prediction = predict(data_dictionary)

    return render_template('predict.html', prediction = prediction)

def predict(dictionary, threshold_opt=0.9375):
    query_df = pd.DataFrame(dictionary, index = [0])
    query_array=np.array(query_df)
    predictions_proba = model.predict_proba(query_array)
    prediction=(predictions_proba[:,1] >= threshold_opt).astype(int)
    return prediction

if __name__ == '__main__':
    app.run(debug=True)
