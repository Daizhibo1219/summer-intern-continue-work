import numpy as np

import os
import urllib
import gzip
import pickle as pickle
#加入anti_label,minus_value
def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='iso-8859-1')
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir, anti_label, minus_value):
    all_data = []
    all_labels = []
    for filename in filenames:        
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)
        print(filename)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)

    print("images shape :", np.shape(images), "type of images :" , type(images))
    print("lables shape :", np.shape(labels), "type of labels :", type(labels))
    width = len(images)
    count_anti = 0
    count_self = 0
    arrys_images = []
    arrys_targets = []
    for i in range(width):
        if labels[i, ] == 1 and count_self < (width*0.1*(1-minus_value)):
            tmp_images = images[i, ]
            arrys_images.append(tmp_images)
            arrys_targets.append(1)
            count_self = count_self + 1
        elif count_anti < (0.1*minus_value*width) and labels[i, ] == anti_label:
            tmp_images = images[i, ]
            arrys_images.append(tmp_images)
            arrys_targets.append(-1)
            count_anti = count_anti + 1
    arrys_targets = np.array(arrys_targets)
    arrys_images = np.array(arrys_images)
    length_proper = int(len(arrys_images) / batch_size) * batch_size
    print(length_proper)
    labels = arrys_targets[0:length_proper]
    images = arrys_images[0:length_proper, :]
    print(type(labels), type(images))
    print(np.shape(labels), np.shape(images))

    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(int(len(images) / batch_size)):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir, anti_label, minus_value):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size,
                        data_dir, anti_label, minus_value),
        cifar_generator(['test_batch'], batch_size, data_dir, anti_label, minus_value)
    )

def SVM_DATA_LOAD(anti_number):
    filenames = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    data_dir = './cifar-10-batches-py'
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)
        print(filename)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    targets = labels
    arrys_images = []
    arrys_targets = []
    print("images长度", len(images))
    print("targets长度", len(targets))
    width = len(images)
    for i in range(width):
        if targets[i,] == 1:
            tmp_images = images[i,]
            arrys_images.append(tmp_images)
            arrys_targets.append(1)

        elif targets[i,] == anti_number:
            tmp_images = images[i,]
            arrys_images.append(tmp_images)
            arrys_targets.append(-1)
    print(type(images), type(images))
    print(np.shape(images), np.shape(targets))
    targets = arrys_targets
    images = arrys_images

    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)

    return (images,targets)

def SVM_TEST_LOAD(anti_number):
    filenames = ['test_batch']
    data_dir = './cifar-10-batches-py'
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(data_dir + '/' + filename)
        all_data.append(data)
        all_labels.append(labels)
        print(filename)

    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    targets = labels
    arrys_images = []
    arrys_targets = []
    print("images长度", len(images))
    print("targets长度", len(targets))
    width = len(images)
    for i in range(width):
        if targets[i,] == 1:
            tmp_images = images[i,]
            arrys_images.append(tmp_images)
            arrys_targets.append(1)

        elif targets[i,] == anti_number:
            tmp_images = images[i,]
            arrys_images.append(tmp_images)
            arrys_targets.append(-1)
    print(type(images), type(images))
    print(np.shape(images), np.shape(targets))
    targets = arrys_targets
    images = arrys_images

    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)

    return (images, targets)