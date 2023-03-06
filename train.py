import os
import keras
from Generator import *
from keras_radam import RAdam
from glob import glob
from utils import logger, cosine_annealing

import tensorflow as tf
num_of_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", num_of_gpus)
if num_of_gpus > 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

# for build resnet18
from keras_resnet import models
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Classification')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--name', default='resnet18', type=str)
    parser.add_argument('--dataset', default='tinyimagenet', type=str)
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--logdir', default='./logs', type=str)
    
    args = parser.parse_args()
    
    run_logdir = logger.get_run_logdir(args.logdir)
    
    input_shape = (224, 224, 3)
    num_classes = 200
    data_path  = '../../datasets/tiny-imagenet-200/'
    if args.dataset.lower() == 'cifar':
        input_shape = (32, 32, 3)
        num_classes = 10
        data_path  = '../../datasets/cifar-10-batches-py/'
    
    
    augments = [iaa.SomeOf((0, 7),
                       [
                           iaa.Identity(),
                           iaa.Rotate((-3,3)),
                           iaa.Sharpen(),
                           iaa.TranslateX(percent=(-0.1, 0.1)),
                           iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.Solarize(),
                           iaa.HistogramEqualization(),
                           iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
                           iaa.ScaleX((0.5, 1.5)),
                       ]),
                            iaa.Sometimes(0.5, iaa.Fliplr(0.5)),
            ]
    if args.dataset.lower() == 'tinyimagenet':
        train_batch = Tiny_imagenet_Generator(batch_size=args.batch,
                                            data_path=data_path,
                                            augs=augments,
                                            is_train=True,
                                            input_shape = input_shape,
                                            num_classes=num_classes
                                            )
        test_batch = Tiny_imagenet_Generator(batch_size=args.batch,
                                            data_path=data_path,
                                            augs=[],
                                            is_train=False,
                                            input_shape = input_shape,
                                            num_classes=num_classes
                                            )
    elif args.dataset.lower() == 'cifar':
        train_batch = Cifar10_Generator(batch_size=args.batch,
                                            data_path=data_path,
                                            augs=augments,
                                            is_train=True,
                                            )
        test_batch = Cifar10_Generator(batch_size=args.batch,
                                            data_path=data_path,
                                            augs=[],
                                            is_train=False,
                                            )
    l2_reg = keras.regularizers.l2(args.weight_decay)
    input_tensor = keras.layers.Input(input_shape)
    
    base_model = models.ResNet18(inputs = input_tensor, classes=num_classes, include_top=False)
    
    x = base_model.output[-1]
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=RAdam(),
                loss=keras.losses.categorical_crossentropy,
                metrics=[keras.metrics.categorical_accuracy]
              )
    
    callbacks = [cosine_annealing.CosineAnnealingScheduler(1e-3, verbose=1),
                    # keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1),
                    keras.callbacks.ModelCheckpoint(filepath='./saved_models/' + args.name + '-' + args.dataset + '_{epoch:05d}.h5',
                                                    verbose=1,
                                                    period=5),
                    keras.callbacks.TensorBoard(run_logdir),
                    # keras.callbacks.EarlyStopping(monitor='loss', patience=15, verbose=1, mode='max')
                ]
    model.fit_generator(train_batch,
                        initial_epoch=0,
                        callbacks=callbacks,
                        epochs=args.epochs,
                        validation_data=test_batch,
                        steps_per_epoch=len(train_batch),
                        validation_steps=len(test_batch),
                        verbose=1)
    
    