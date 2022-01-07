import pandas as pd
import pickle
import math
import numpy as np
from flask import Flask, render_template, request

variables = ['num_C_YEAR', 'C_MNTH', 'C_WDAY','C_HOUR', 'num_C_VEHS', 'num_Random', 'ohe_x0_01', 'ohe_x0_02', 'ohe_x0_03',
       'ohe_x0_04', 'ohe_x0_QQ', 'ohe_x0_None', 'ohe_x1_01', 'ohe_x1_02',
       'ohe_x1_QQ', 'ohe_x1_None', 'ohe_x2_01', 'ohe_x2_02', 'ohe_x2_Q',
       'ohe_x2_None', 'ohe_x3_01', 'ohe_x3_02', 'ohe_x3_Q', 'ohe_x3_None',
       'ohe_x4_01', 'ohe_x4_02', 'ohe_x4_Q', 'ohe_x4_None', 'ohe_x5_01',
       'ohe_x5_03', 'ohe_x5_QQ', 'ohe_x5_None']

#create an instance of Flask
app = Flask(__name__)

with open('/Users/aitor/Desktop/Máster Ciencia de Datos/Aprendizaje automático/Machine_Learning/big_practice/Models/RF.pickle', 'rb') as f:
    model=pickle.load(f)

# cuando el usuario acceda a la url "/", automáticamente se iniciará la función index().
# la función index() se encargará de devolvernos nuestra html principal crashes.html.
# en nuestro caso crashes.html servirá para que el usuario introduzca los datos.
# render_template se encarga de buscar el html crashes en nuestra carpeta de templates y genera un html a partir de dicho archivo.
@app.route('/')# usamos el método por defecto GET (Queremos extraer datos del html)
def home():
    return render_template("crashes.html")

# ahora creamos otra instancia de @app.route. En este caso, cuando el usuario acceda a "/predict", se iniciará la función
# predict(). Esta función predict() se encargará de tomar el input proporcionado por el usuario y transformarlo de tal manera
# que pueda ser utilizado para predecir con nuestro modelo previamente cargado.
@app.route('/predict/',methods=["POST"])#POST nos sirve para introducir datos en el html
def predict():
    data_dictionary = dict()
    for variable in variables:
        if variable == "C_MNTH":
            month = int(request.form.get(variable))
            print(month)
            NA_cos_C_MNTH = np.cos(2*math.pi*month/12)
            NA_sin_C_MNTH = np.sin(2*math.pi*month/12)
            data_dictionary["NA_cos_C_MNTH"] = NA_cos_C_MNTH
            data_dictionary["NA_sin_C_MNTH"] = NA_sin_C_MNTH
        elif variable == "C_HOUR":
            hour = int(request.form.get(variable))
            print(hour)
            NA_cos_C_HOUR = np.cos(2*math.pi*hour/24)
            NA_sin_C_HOUR = np.sin(2*math.pi*hour/24)
            data_dictionary["NA_cos_C_HOUR"] = NA_cos_C_HOUR
            data_dictionary["NA_sin_C_HOUR"] = NA_sin_C_HOUR
        elif variable == "C_WDAY":
            wday = int(request.form.get(variable))
            print(wday)
            NA_cos_C_WDAY = np.cos(2*math.pi*wday/7)
            NA_sin_C_WDAY = np.sin(2*math.pi*wday/7)
            data_dictionary["NA_cos_C_WDAY"] = NA_cos_C_WDAY
            data_dictionary["NA_sin_C_WDAY"] = NA_sin_C_WDAY
        else:
            data_dictionary[variable] = request.form.get(variable)
    query_df = pd.DataFrame(data_dictionary, index = [0])
    prediction = preprocess_and_predict_data(query_df)

    return render_template('predict.html', prediction = prediction)

def preprocess_and_predict_data(df):
    test_data = df
    test_data = np.array(test_data)
    print(test_data)
    prediction = model.predict(test_data)
    return prediction

if __name__ == '__main__':
    app.run(debug=True)
