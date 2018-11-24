# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 22:44:58 2018

@author: Computer
"""

# Cargando modelo de disco
import tensorflow as tf
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import numpy as np

def cargarModelo():
    json_file = open('./model/classifier.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./model/classifier.h5")
    print("Cargando modelo desde el disco ...")
    loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    print("Modelo cargado de disco!")
    graph = tf.get_default_graph()
    return loaded_model, graph