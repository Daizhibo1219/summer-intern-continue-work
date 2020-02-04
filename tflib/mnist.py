import numpy

import os
import urllib
import gzip
import pickle as pickle

def mnist_generator(data, batch_size, n_labelled, limit=None):
    images, targets = data
    #尝试只取1
    '''
    arrys_images=[]
    arrys_targets=[]
    print("images长度", len(images))
    print("targets长度",len(targets))
    width = len(images)
    for i in range(width):
        if targets[i, ] == 1:
            tmp_images = images[i, ]
            arrys_images.append(tmp_images)
            arrys_targets.append(1)
    arrys_targets = numpy.array(arrys_targets)
    arrys_images = numpy.array(arrys_images)
    length_proper = int(len(arrys_images)/batch_size) * batch_size
    print(length_proper)
    targets = arrys_targets[0:length_proper]
    images = arrys_images[0:length_proper, :]
    print(type(images), type(images))
    print(numpy.shape(images), numpy.shape(targets))
    '''
    #尝试结束
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(images)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(targets)
    if limit is not None:
        print ("WARNING ONLY FIRST {} MNIST DIGITS".format(limit))#改了print
        images = images.astype('float32')[:limit]
        targets = targets.astype('int32')[:limit]
    if n_labelled is not None:
        labelled = numpy.zeros(len(images), dtype='int32')
        labelled[:n_labelled] = 1

    def get_epoch():
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(images)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(targets)

        if n_labelled is not None:
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labelled)

        image_batches = images.reshape(-1, batch_size, 784)
        target_batches = targets.reshape(-1, batch_size)

        if n_labelled is not None:
            labelled_batches = labelled.reshape(-1, batch_size)

            for i in range(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]), numpy.copy(labelled))

        else:

            for i in range(len(image_batches)):
                yield (numpy.copy(image_batches[i]), numpy.copy(target_batches[i]))

    return get_epoch

def load(batch_size, test_batch_size, n_labelled=None):
    filepath = 'E:/python_test/mnist.pkl.gz'#改了文件路径
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print ("Couldn't find MNIST dataset in /tmp, downloading...")#改了print
        urllib.request.urlretrieve(url, filepath)#加了request

    with gzip.open('E:/python_test/mnist.pkl.gz', 'rb') as f:#改了文件路径
        train_data, dev_data, test_data = pickle.load(f, encoding='bytes')#加了, encoding='bytes'
    '''
    尝试只把‘1’取出来
    '''
    #print(numpy.shape(train_data))

    return (
        mnist_generator(train_data, batch_size, n_labelled),
        mnist_generator(dev_data, test_batch_size, n_labelled),
        mnist_generator(test_data, test_batch_size, n_labelled)
    )