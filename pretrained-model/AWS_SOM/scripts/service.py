# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 22:30:38 2018

@author: Computer
"""

#Import Flask
from flask import Flask, request
from cnn_executor import cargarModelo
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Initialize the application service
app = Flask(__name__)
global loaded_model, graph
loaded_model, graph = cargarModelo()

#Define a route
@app.route('/')
def main_page():
	return 'Bienvenido Clasificacion de opinion de magistrados'

@app.route('/AWS_SOM/', methods=['GET','POST'])
def rayosx():
	return 'clasificacion de opinion de magistrados!'

@app.route('/AWS_SOM/default/', methods=['GET','POST'])
def default():
    #print (request.args)
    #test_image_path = './dataset/test.csv'
    from sklearn.externals import joblib
    file_scale = "./model/classifier.save"
    # 1.1 Normalizando campo de Calculo
    sc = joblib.load(file_scale)
    opentest = pd.read_csv(r'dataset/test.csv',usecols=['Edificio','P1a','P2','P3','P4','P5a','P5b','P5c','P5d','P6','P7','P8','P9','P10a','P10b','P10c','P10d','P11','P12','P13','P14','P15','P16'])
    ##aplicar en la nueva data
    transform_test=sc.transform(opentest) 
    result = loaded_model.predict(transform_test)
    print (result)
    for i in range(len(result)):
        if result[i] > 0.5:
            print(result[i], ' --> A favor')
        else:
            print(result[i], ' --> En contra ')
    return result
# Run de application
app.run(host='0.0.0.0',port=5100) 