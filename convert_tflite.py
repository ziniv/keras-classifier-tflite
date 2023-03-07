import os
import tensorflow as tf
import cv2
import glob
import random
import numpy as np
import argparse

def get_resnet18(input_shape, num_classes, name):
    import keras
    from keras_resnet import models
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D
    input_tensor = keras.layers.Input(input_shape)
    if name == 'resnet18':
        base_model = models.ResNet18(inputs = input_tensor, classes=num_classes, include_top=False)
        x = base_model.output[-1]
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
    
    

if __name__ == "__main__":
    from tensorflow import keras
    parser = argparse.ArgumentParser(description="convert h5 to tflite")
    parser.add_argument('--name', default='resnet18', type=str)
    parser.add_argument('--input', default='./saved_models/resnet18-cifar_final.h5', type=str)
    parser.add_argument('--output', default='./saved_models/', type=str)
    parser.add_argument('--is_quan', default=True, type=bool)
    
    args = parser.parse_args()
    
    filename = args.input.split('/')[-1].split('.h5')[0]
    restore_filename = os.path.join(args.output, filename+'_restore.h5')
    output_filename = os.path.join(args.output, filename+'.tflite')
    
    model = get_resnet18((32, 32, 3), 10, 'resnet18')
    model.load_weights(args.input)
    model.save(restore_filename)
    
    model = keras.models.load_model(restore_filename)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(output_filename, 'wb') as f:
        f.write(tflite_model)
    
    