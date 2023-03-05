import glob
import os
import cv2
import numpy as np
import tqdm
from imgaug import augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch
from tensorflow import keras

from print_color import bcolors

class Cifar10_Generator(keras.utils.Sequence):
    def __init__(self, batch_size, input_shape, num_classes, data_path, augs, expand_path=None, is_train=True):
        """

        :param batch_size:
        :param input_shape:
        :param num_classes:
        :param data_path:
        :param augs:
        :param is_train:
        """
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.is_train = is_train

        self.label_list = ['negative', 'positive']
        if is_train:
            self.data = []
            for data_file in data_path:
                self.data += glob.glob(data_file + '/train/*/**.jpg')
            if len(expand_path) > 0:
                for expand_file in expand_path:
                    expand_datas = glob.glob(expand_file + '/train/*/**.jpg')
                    expand_len = int(len(expand_datas)*0.3)
                    self.data += expand_datas[:expand_len]
            self.train_list = dict()
            for data in self.data:
                norm_data = os.path.normpath(data)
                label = norm_data.split(os.path.sep)[-2]
                self.train_list[data] = self.label_list.index(label)

        else:
            self.data = []
            for data_file in data_path:
                if not os.path.isdir(data_file + '/valid/'):
                    self.data += glob.glob(data_file + '/test/*/**.jpg')
                else:
                    self.data += glob.glob(data_file + '/valid/*/**.jpg')
            
            self.val_list = dict()
            for data in self.data:
                norm_data = os.path.normpath(data)
                label = norm_data.split(os.path.sep)[-2]
                self.val_list[data] = self.label_list.index(label)
        self.augmenter = iaa.Sequential(augs)

        self.indexes = None
        self.step = 0
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
                label = self.val_list[img_file]
            cls.append(label)

        batch = UnnormalizedBatch(images=imgs, data=cls)
        augmented_data = list(self.augmenter.augment_batches(batch, background=False))

        for i in range(len(data_list)):
            img = augmented_data[0].images_aug[i]
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            ## TEST img
            # img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
            # cv2.imshow('test', img)
            # cv2.imshow('unaug', augmented_data[0].images_unaug[i])
            # cv2.waitKey(0)
            ###
            img = img.astype(np.float) / 255.
            label = augmented_data[0].data[i]
            batch_images[i] = img
            # print(label, data_list[i])
            label = keras.utils.to_categorical(label, num_classes=self.num_classes)
            batch_cls[i] = label

        return batch_images, batch_cls

class Tiny_imagenet_Generator(keras.utils.Sequence):
    def __init__(self, batch_size, input_shape, num_classes, data_path, augs, is_train=True, label_smoothing=False):
        """
        :param batch_size:
        :param input_shape:
        :param output_stride:
        :param num_classes:
        :param data_path:
        :param augs:
        :param is_train:
        """
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
            print(label, data_list[i])
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



def unpacking_cifar10(path):
    import numpy as np
    from PIL import Image
    import pickle
    import cv2
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    meta_data = os.path.join(path, 'batches.meta')
    batch_data = glob.glob(os.path.join(path, 'data_batch*'))
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')

    with open(meta_data, 'rb') as infile:
        data = pickle.load(infile, encoding='latin1')
        classes = data['label_names']
    
    for batch in batch_data:
        print('Unpacking Train File {}'.format(batch))
        train_file = unpickle(batch)

        train_data = train_file[b'data']
        train_data_reshape = np.vstack(train_data).reshape((-1, 3, 32, 32))
        train_labels = train_file[b'labels']
        train_filename = train_file[b'filenames']
        print(len(train_data_reshape))
        for i in train_data_reshape:
            print(i.shape)
            train_img = i.swapaxes(0, 2)
            print(train_img.shape)
            cv2.imshow("dd", train_img)
            if cv2.waitKey(0) == ord('q'):
                break



if __name__ == '__main__':

    # unpacking_cifar10('../dataset/cifar-10-batches-py')

    NumOfClasses = 200
    input_shape = (224, 224, 3)
    augument_crop_w = 224
    augument_crop_h = 224
    train_batch_size = 1
    tinyimagenet_path = '../dataset/tiny-imagenet-200/'

    augments = [iaa.Sometimes(0.5,
                       [
                           # iaa.Crop(px=(0, 0, 6, 0))
                           iaa.Affine(scale=1.2)
                       ])
            ]

    bgen = Tiny_imagenet_Generator(batch_size=train_batch_size, input_shape=input_shape,
                                                    num_classes=NumOfClasses,
                                                    data_path=tinyimagenet_path,
                                                    augs=augments
                                                    )

    while True:
        for i in tqdm.tqdm(range(bgen.__len__())):
            bgen.__getitem__(i)
        bgen.on_epoch_end()

    
