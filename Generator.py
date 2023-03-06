import os
import pickle
import glob
import cv2
import numpy as np
import tqdm
from imgaug import augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch
from tensorflow import keras


class Cifar10_Generator(keras.utils.Sequence):
    def __init__(self, batch_size, data_path, augs, is_train=True, label_smoothing=False):
        meta_datapath = os.path.join(data_path, 'batches.meta')
        batch_datapath = glob.glob(os.path.join(data_path, 'data_batch*'))
        test_datapath = glob.glob(os.path.join(data_path, 'test_batch'))
        
        self.batch_size = batch_size
        self.input_shape = (32, 32, 3)
        self.num_classes = 10
        self.is_train = is_train
        self.label_smoothing = label_smoothing
        self.eps = 0.1
        self.augmenter = iaa.Sequential(augs)
        self.classes = []
        with open(meta_datapath, 'rb') as meta_fo:
            data = pickle.load(meta_fo, encoding='latin1')
            self.classes = data['label_names']
        self.data_list, self.label_list = self.unpacking_data(batch_datapath if self.is_train else test_datapath)
        self.indexes = None
        self.on_epoch_end()
    
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def unpacking_data(self, file_list):
        data_list = []
        label_list = []
        for i, data in enumerate(file_list):
            train_file = self.unpickle(data)
            train_data = train_file[b'data']
            train_data_reshape = np.vstack(train_data).reshape((-1, 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2)
            train_labels = train_file[b'labels']
            data_list = train_data_reshape if i == 0 else np.concatenate((data_list, train_data_reshape), axis=0)
            label_list = train_labels if i == 0 else np.concatenate((label_list, train_labels), axis=0)
        return data_list, label_list

    def __len__(self):
        return int(np.floor(len(self.data_list) / self.batch_size))

    def __getitem__(self, item):
        indexes = self.indexes[item * self.batch_size: (item + 1) * self.batch_size]
        data_list = [self.data_list[i] for i in indexes]
        x, y = self.__data_gen(data_list)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_list))
        if self.is_train:
            np.random.shuffle(self.indexes)

    def __data_gen(self, data_list):
        cv2.setNumThreads(0)
        batch_images = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                dtype=np.float32)
        if self.label_smoothing:
            batch_cls = np.ones(shape=(self.batch_size, self.num_classes), dtype=np.float32) * self.eps / (
                    self.num_classes - 1)
        else:
            batch_cls = np.zeros(shape=(self.batch_size, self.num_classes), dtype=np.float32)
        imgs = []
        cls = []
        for i, img_data in enumerate(data_list):
            imgs.append(img_data)
            label = self.label_list[self.indexes[i]]
            cls.append(label)

        batch = UnnormalizedBatch(images=imgs, data=cls)
        augmented_data = list(self.augmenter.augment_batches(batch, background=False))

        for i in range(len(data_list)):
            img = augmented_data[0].images_aug[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float) / 255.
            label = augmented_data[0].data[i]
            batch_images[i] = img
            if self.label_smoothing:
                batch_cls[i, label] = 1.0 - self.eps
            else:
                label = keras.utils.to_categorical(label, num_classes=self.num_classes)
                batch_cls[i] = label
            # cv2.imshow('test', img)
            # cv2.imshow('unaug', augmented_data[0].images_unaug[i])
            # if cv2.waitKey(0) == ord('q'):
            #     break

        return batch_images, batch_cls

class Tiny_imagenet_Generator(keras.utils.Sequence):
    def __init__(self, batch_size, input_shape, num_classes, data_path, augs, is_train=True, label_smoothing=False):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.is_train = is_train
        self.label_smoothing = label_smoothing
        self.eps = 0.1
        self.augmenter = iaa.Sequential(augs)

        with open(data_path + '/wnids.txt', 'r') as f:
            self.label_list = f.read().splitlines()
            
        if is_train:
            self.data = glob.glob(data_path + '/train/*/images/*.JPEG')
            self.train_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-3]
                self.train_list[data] = self.label_list.index(label)
                
        else:
            self.data = glob.glob(data_path + '/val/images/*.JPEG')
            self.val_list = dict()
            with open(data_path + '/val/val_annotations.txt', 'r') as f:
                val_labels = f.read().splitlines()
                for label in val_labels:
                    f_name, label, _, _, _, _ = label.split('\t')
                    self.val_list[f_name] = self.label_list.index(label)

        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, item):
        indexes = self.indexes[item * self.batch_size: (item + 1) * self.batch_size]
        data_list = [self.data[i] for i in indexes]
        x, y = self.__data_gen(data_list)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))
        if self.is_train:
            np.random.shuffle(self.indexes)

    def __data_gen(self, data_list):
        cv2.setNumThreads(0)
        batch_images = np.zeros(shape=(self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                                dtype=np.float32)

        if self.label_smoothing:
            batch_cls = np.ones(shape=(self.batch_size, self.num_classes), dtype=np.float32) * self.eps / (
                    self.num_classes - 1)
        else:
            batch_cls = np.zeros(shape=(self.batch_size, self.num_classes), dtype=np.float32)

        imgs = []
        cls = []
        for img_file in data_list:
            img = cv2.imread(img_file)
            img = cv2.resize(img, dsize=(self.input_shape[1], self.input_shape[0]))
            imgs.append(img)
            # cls append
            if self.is_train:
                label = self.train_list[img_file]
            else:
                label = self.val_list[os.path.basename(img_file)]
            cls.append(label)

        batch = UnnormalizedBatch(images=imgs, data=cls)
        augmented_data = list(self.augmenter.augment_batches(batch, background=False))

        for i in range(len(data_list)):
            img = augmented_data[0].images_aug[i]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float) / 255.
            label = augmented_data[0].data[i]
            batch_images[i] = img
            if self.label_smoothing:
                batch_cls[i, label] = 1.0 - self.eps
            else:
                label = keras.utils.to_categorical(label, num_classes=self.num_classes)
                batch_cls[i] = label
            cv2.imshow('test', img)
            cv2.imshow('unaug', augmented_data[0].images_unaug[i])
            if cv2.waitKey(0) == ord('q'):
                break

        return batch_images, batch_cls


if __name__ == '__main__':
    '''
    1. tinyimagenet
        - classes = 200
        - input_shape = 224, 224, 3 (commonly)
        - image_path = '../../datasets/tiny-imagenet-200/'
    2. cifar10
        - classes = 10
        - input_shape = 32, 32, 3 (fix)
        - image_path = '../../datasets/cifar-10-batches-py/'
    '''
    train_batch_size = 1
    cifar_path = '../../datasets/cifar-10-batches-py/'

    augments = [iaa.Sometimes(0.5,
                       [
                           # iaa.Crop(px=(0, 0, 6, 0))
                           iaa.Affine(scale=1.2)
                       ])
            ]

    bgen = Cifar10_Generator(batch_size=train_batch_size,
                             data_path=cifar_path,
                             augs=augments
                             )

    while True:
        for i in tqdm.tqdm(range(bgen.__len__())):
            bgen.__getitem__(i)
        bgen.on_epoch_end()

    
