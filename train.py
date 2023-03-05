import os
import tensorflow as tf
num_of_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", num_of_gpus)
if num_of_gpus > 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras
from Generator import Tiny_imagenet_Generator
from keras_radam import RAdam
from glob import glob
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Train Classification')
    args.add_argument('--epochs', default=300, type=int)
    args.add_argument('--name', default='resnet18', type=str)
    args.add_argument('--dataset', default='tinyimagenet', type=str)
    
    