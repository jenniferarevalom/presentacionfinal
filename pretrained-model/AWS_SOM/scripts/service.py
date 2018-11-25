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
from sklearn.externals import joblib

#Initialize the application service
app = Flask(__name__)
global loaded_model, graph
loaded_model, graph = cargarModelo()

#Define a route
@app.route('/')
def main_page():
	return 'Bienvenido a la Presentacion Final de Curso Redes Neuronales '

@app.route('/AWS_SOM/', methods=['GET','POST'])
def rayosx():
	return 'Modelo de Clasificacion SOM acerca de Opinion de Magistrados'

@app.route('/AWS_SOM/default/', methods=['GET','POST'])
def default():
    #print (request.args)
   	# Show
    image_name = request.args.get("dataluz")
    img_path='../dataset/'+image_name+'.csv'
    #test_image_path = './dataset/'+test+'.csv'
    opentest = pd.read_csv(img_path)
    file_scale = "../model/classifier.save"
    # 1.1 Normalizando campo de Calculo
    sc = joblib.load(file_scale)
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