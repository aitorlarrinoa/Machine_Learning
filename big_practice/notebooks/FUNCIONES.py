#!/usr/bin/env python
# coding: utf-8

# # ANÁLISIS DE SINIESTRALIDAD

# ### CUNEF MUCD (2021/22)

# - Aitor Larriona Rementería
# - Diego Cendán Bedregal

# # FUNCIONES

# En este notebook estarán presentes las funciones de las que haremos uso durante los notebooks 01 y 02. Cada una de las funciones vendrá con una breve explicación la cual nos ayudará a comprender el funcionamiento de la misma.

# # LIBRERÍAS

import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
import time
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, accuracy_score, f1_score, recall_score
import scikitplot as skplt

# ## FUNCIONES PARA NOTEBOOK 01 (EDA)

# La función *valores_unicos* nos será útil para ver qué valores únicos aparecen en cada una de las columnas de un dataset. Estos valores únicos apareceran en forma de diccionario en el que las claves son las columnas y los valores son una lista con los valores únicos de esa columna

# Creamos una función en la que introducimos un dataset y nos devuelve un diccionario en el cual las claves son las columnas y los valores son los valores únicos de dichas columnas. Esto nos servirá para tener una idea de los valores que toman las columnas en nuestro dataset

# In[3]:


def valores_unicos(dataset, k=50):
    dict_valores_unicos = dict()
    for columna in dataset:
        if len(list(dataset[columna].unique())) < k:
            dict_valores_unicos[columna] = list(dataset[columna].unique())
    return dict_valores_unicos


# La funcion *unificar* elimina los valores repetidos de cada columna y transforma las columnas elegidas a númericas. 

# En primer lugar, creamos una función que nos permite modificar el dataset de tal forma que no tengamos valores repetidos. A partir de ahora, los valores de nuestro dataset serán de la forma "01", "02" y así sucesivamente. Posteriormente convertimos las columnas a númericas. Para ello, si si la funcion encuentra algún valor que no sea convertible a valor entero y diferente de NaN, lo considerará como valor nulo.

# In[4]:


def unificar(dataset, columns_to_numeric):
    list_values=list()
    for column in dataset.drop(columns_to_numeric, axis=1):
        if dataset[column].dtype != int and column != 'P_SEX':
            for value in dataset[column]:
                if not pd.isnull(value) and value != "Q":
                    if value in ["1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0"]:
                        value = int(float(value))
                        valor_nuevo = str(value).zfill(2)
                        list_values.append(valor_nuevo)
                    else:
                        valor_nuevo = str(value).zfill(2)
                        list_values.append(valor_nuevo)
                elif pd.isnull(value):
                    list_values.append(np.nan)     
                else:
                    list_values.append("Q")
            # introducimos los valores introducidos en la lista a la columna del dataset
            dataset[column] = list_values
            # nos olvidamos de los valores de la lista y la inicializamos vacía otra vez para la siguiente iteración
            list_values=list()
    # pasamos las columnas que queremos a numéricas en las que si se encuentra algún valor diferente a algo convertible
    # a entero y diferente de nan, entonces será considerado como nan (esto es gracias a errors="coerce").
    dataset[columns_to_numeric] = dataset[columns_to_numeric].apply(pd.to_numeric, errors="coerce")
    for column in columns_to_numeric:
        if dataset[column].dtype==float:
            dataset[column] = dataset[column].astype('Int64')
    return dataset


# La función *codifiacion_ciclica* modifica las variables para convertirlas en cíclias, es decir, que los datos se repiten de manera regular cada X tiempo. De esta forma indicamos a nuestro modelo que ningún dato es mayor o menor al anterior si no que se comportan de forma cíclica.  Para ello haremos uso de las funciones coseno y seno para codificar las variables seleccionadas.

# In[5]:


def codificacion_ciclica(dataset, columns):
    for columna in columns:
        dataset[columna+"_norm"] = 2*math.pi*dataset[columna]/dataset[columna].max()
        dataset["cos_"+columna] = np.cos(dataset[columna+"_norm"])
        dataset["sin_"+columna] = np.sin(dataset[columna+"_norm"])
        dataset = dataset.drop([columna+"_norm"], axis=1)
    return dataset


# La funcion norm_category agrupa cada variable categorica según el valor que pueda tomar nuestra variabla Target. Finalmente suma cuantos valores tenemos según cada categoría 

# In[6]:


def norm_category(df = None, obj_val = "", cat_val = ""):
    grouped = df.groupby([obj_val, cat_val]).count().iloc[:,1]
    grouped = grouped.reset_index()
    grouped.columns = [obj_val, cat_val, 'counted']
    grouped['group%'] = np.round(grouped['counted'] / 
                                 grouped.groupby(obj_val)['counted'].transform('sum') * 100, decimals = 3)
    return grouped


# La función *graf_barras* realiza un gráfico de barras para cada una de las variables siempre y cuando sean consideradas variables enteras  o categoricas. Por otra parte, realiza un boxplot o gráfico de cajas para cada una de las variables continuas de nuestro dataset. Todos los gráficos se realizan en funcion de nuestra variable target. 
# La función nos mostrará también el número de valores nulos existentes en cada columna. 

# In[7]:


def graf_barras(df, columna, isContinuous, var_objetivo):
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    
    count_null = df[columna].isnull().sum()
    if isContinuous:
        
        sns.histplot(df.loc[df[columna].notnull(), columna], kde=False, ax=ax1)
    else:
        sns.countplot(df[columna].dropna(), order=sorted(df[columna].dropna().unique()), color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(columna)
    ax1.set_ylabel('Count')
    ax1.set_title(columna+ ' Numero de nulos: '+str(count_null))
    plt.xticks(rotation = 90)


    if isContinuous:
        sns.boxplot(x=columna, y=var_objetivo, data=df, ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title(columna + ' by '+var_objetivo)
    else:
        data = df.groupby(columna)[var_objetivo].value_counts(normalize=True).to_frame('proportion').reset_index() 
        data.columns = [columna, var_objetivo, 'proportion']
        data.columns = [columna, var_objetivo, 'proportion']
        #sns.barplot(x = col_name, y = 'proportion', hue= target, data = data, saturation=1, ax=ax2)
        sns.barplot(x = columna, y = 'proportion', hue= var_objetivo, data = data, saturation=1, ax=ax2)
        ax2.set_ylabel(var_objetivo+' fraction')
        ax2.set_title(var_objetivo)
        plt.xticks(rotation = 90)
    ax2.set_xlabel(columna)
    
    plt.tight_layout()


# ## FUNCIONES PARA NOTEBOOK 02 (PREPARACIÓN DE DATOS)

# La función valores_unicos (también definida en el notebook *01_EDA*) sirve para obtener una idea de los valores únicos que hay en cada variable de un dataset. El resultado se mostrará mediante un diccionario en el que las claves son las columnas y los valores son listas en las que se muestran los valores únicos que tiene dicha variable

# In[8]:


def valores_unicos(dataset, k=50):
    valores_unicos = dict()
    for columna in dataset:
        if len(list(dataset[columna].unique())) < k:
            valores_unicos[columna] = list(dataset[columna].unique())
    return valores_unicos


# La función reemplazar_NA_moda nos permitirá reemplazar los valores nulos por la moda. Finalmente decicidimos realizar la imputación de valores missing mediante PipeLines.

# In[9]:


def reemplazar_NA_moda(dataset):
    for columna in dataset:
        if sum(pd.isnull(dataset[columna]))>1:
            dataset.fillna(mode(dataset[columna].dropna()), inplace=True)
    return dataset  


# La función *reemplazar_Nas* nos permitirá reemplazar los valores nulos por un valor aleatorio de la columna siempre en función de cada valor de la variable objetivo. Finalmente decidimos realizar la imputación de valores missing mediante PipeLines.

# In[10]:


def reemplazar_NAs(dataset, target, columns_mode, var_referencia):
    for columna in dataset.drop(columns_mode, axis=1):
        if dataset[columna].isna().any() and columna != var_referencia:
            a = dataset[columna].isnull().sum()
            b=0
            for i in dataset.index:
                if pd.isnull(dataset[columna][i]):
                    dataset[columna][i] = random.choice(list(dataset[dataset[var_referencia] == dataset[var_referencia][i]][columna].dropna()))
                    b+=1
                    print(b/a)
    for columna in dataset[columns_mode]:
        if sum(pd.isnull(dataset[columna]))>1 and columna != target:
            dataset.fillna(mode(dataset[columna].dropna()), inplace=True)
    return dataset 


# ## FUNCIONES PARA EL NOTEBOOK 03(MODELOS)

# La función medidas es una función que nos permitirá conocer las métricas f1, accuracy y roc_score de cada modelo que vayamos obteniendo. Todo ello será presentado en un data frame de una única fila y 3 columnas

# In[11]:


def medidas(ytest, predictions, predictions_proba):
    ROC_score = list()
    F1_1 = list()
    acc_1 = list()
    recall=list()
    roc_score = roc_auc_score(ytest, predictions_proba[:,1])
    ROC_score.append(roc_score)
    f1 = f1_score(ytest, predictions)
    F1_1.append(f1)
    ac = accuracy_score(ytest, predictions)
    acc_1.append(ac)
    rec = recall_score(ytest, predictions)
    recall.append(rec)
    data = {'F1': F1_1, "accuracy" :acc_1, 'roc_score':ROC_score, "recall":recall}

    return pd.DataFrame(data, columns=["F1", 'roc_score', 'accuracy', "recall"])


# La función modelos nos permitirá entrenar un modelo y hallar predicciones del mismo. Esta función será útil para encapsular código y no tener que escribir la misma estructura para todos los modelos que se calcularán. Además, esta función también nos devolverá el tiempo de computo para el modelo

# In[12]:


def modelos(xtrain, ytrain, xtest, modelo, ytest):
    start=time.time()
    model = modelo
    model_train=model.fit(xtrain, ytrain)
    predictions = model.predict(xtest)
    predictions_proba = model.predict_proba(xtest)
    end=time.time()
    
    # matrices de cinfusión absoluta y normalizada
    cm = confusion_matrix(ytest, predictions)
    cm_norm = confusion_matrix(ytest, predictions, normalize="true")

    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=1, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    plt.title("Confusion matrix")

    plt.figure(figsize=(5,5))
    sns.heatmap(cm_norm, annot=True, fmt=".3f", linewidths=1, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    plt.title("Normalized confusion matrix")
    
    # curva roc
    plt.figure(figsize=(5,3))
    yhat = predictions_proba[:, 1]
    fpr, tpr, thresholds = roc_curve(ytest, yhat)
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='RF PCA')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    
    # curva de ganancia
    skplt.metrics.plot_cumulative_gain(ytest, predictions_proba)
    plt.show()
    return start, end, predictions, predictions_proba, model_train


# In[13]:


def get_feature_names(column_transformer):
    def get_names(trans):
        if trans == 'drop' or (hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            if column is None:
                return []
            else:
                return [name + "_" + f for f in column]

        return [name + "_" + f for f in trans.get_feature_names()]
    feature_names = []
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        l_transformers = list(column_transformer._iter(fitted=True))
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            _names = get_feature_names(trans)
            if len(_names)==0:
                _names = [name + "_" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names


# ## FUNCIONES PARA EL NOTEBOOK 04(INTERPRETABILIDAD)

# In[13]:


def get_feature_names(column_transformer):
    def get_names(trans):
        if trans == 'drop' or (hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            if column is None:
                return []
            else:
                return [name + "_" + f for f in column]

        return [name + "_" + f for f in trans.get_feature_names()]
    feature_names = []
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        l_transformers = list(column_transformer._iter(fitted=True))
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            _names = get_feature_names(trans)
            if len(_names)==0:
                _names = [name + "_" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names

