import tensorflow.keras as keras
import tensorflow as tf
from keras_radam import RAdam
import cv2
import glob
import random
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert h5 to tflite")
    parser.add_argument('--name', default='resnet18', type=str)
    parser.add_argument('--input', default='./saves_models/resnet18-cifar_final.h5', type=str)
    parser.add_argument('--output', default='./saves_models/', type=str)
    parser.add_argument('--is_quan', default=True, type=bool)
    
    args = parser.parse_args()
    
    model = keras.models.load_model(args.input)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    filename = args.input.split('/')[-1].split('.h5')[0]
    print(filename)
    # with open(args)
    