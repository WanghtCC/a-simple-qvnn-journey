import os
import os.path
import struct
import gzip
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
sys.path.insert(0, '../dataset')

try:
    matplotlib.use('TkAgg')
except: # pylint: disable=bare-except
    print('Can not use matplot TkAgg')
    pass

try:
    import urllib.request
except ImportError:
    raise ImportError('You should use python 3.x')

url_base = 'file:/Users/wanght/Documents/微电子研究院/科研/dataset/MNIST/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

def _download(dataset_dir, file_name):
    '''
    Download function for mnist dataset file
    '''
    file_path = dataset_dir + "/" + file_name
    if os.path.exists(file_path):
        with gzip.GzipFile(file_path) as file:
            file_path_ungz = file_path[:-3].replace("\\", "/")
            if not os.path.exists(file_path_ungz):
                open(file_path_ungz, 'wb').write(file.read())
        return
    
    print("Downloading " + file_name + " ...")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    if os.path.exists(file_path):
        with gzip.GzipFile(file_path) as file:
            file_path_ungz = file_path[:-3].replace("\\", "/")
            file_path_ungz = file_path_ungz.replace("-idx", ".idx")
            if not os.path.exists(file_path_ungz):
                open(file_path_ungz, 'wb').write(file.read())
    print("Done")

def download_mnist(dataset_dir):
    for v in key_file.values():
        _download(dataset_dir, v)

if not os.path.exists("./result"):
    os.mkdir("./result")
else:
    pass

def load_mnist(dataset='training_data', digits=np.arange(10), path="./src/dataset"):
    '''
    Load mnist dataset
    '''
    from array import array as pyarray
    download_mnist(path)
    if dataset == "training_data":
        fname_img = os.path.join(path, "train-images.idx3-ubyte").replace("\\", "/")
        fname_label = os.path.join(path, "train-labels.idx1-ubyte").replace("\\", "/")
    elif dataset == "testing_data":
        fname_img = os.path.join(path, "t10k-images.idx3-ubyte").replace("\\", "/")
        fname_label = os.path.join(path, "t10k-labels.idx1-ubyte").replace("\\", "/")
    else:
        raise ValueError("dataset must be 'testing_data' or 'training_data'")
    
    flbl = open(fname_label, 'rb')
    _, size = struct.unpack(">II", flbl.read(8))

    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)
    images = np.zeros((N, rows, cols))
    labels = np.zeros((N, 1), dtype=int)
    for i in range(len(ind)):
        images[i] = np.asarray(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
        
    return images, labels

def load_mnist_5_5(dataset='training_data', digits=np.arange(10), path="./src/dataset"):
    '''
    Load mnist dataset
    '''
    from array import array as pyarray
    download_mnist(path)
    if dataset == "training_data":
        fname_img = os.path.join(path, "train-images.idx3-ubyte").replace("\\", "/")
        fname_label = os.path.join(path, "train-labels.idx1-ubyte").replace("\\", "/")
    elif dataset == "testing_data":
        fname_img = os.path.join(path, "t10k-images.idx3-ubyte").replace("\\", "/")
        fname_label = os.path.join(path, "t10k-labels.idx1-ubyte").replace("\\", "/")
    else:
        raise ValueError("dataset must be 'testing_data' or 'training_data'")
    
    flbl = open(fname_label, 'rb')
    _, size = struct.unpack(">II", flbl.read(8))

    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)
    images = np.zeros((N, rows, cols))
    labels = np.zeros((N, 1), dtype=int)
    new_images = np.zeros((N, 4, 4))
    for i in range(len(ind)):
        images[i] = np.asarray(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols]).reshape((rows, cols))
        new_images[i] = cv2.resize(images[i], (4, 4), interpolation=cv2.INTER_AREA)
        labels[i] = lbl[ind[i]]
    return new_images, labels


def load_mnist_resize(dataset='training_data', digits=np.arange(10), path="./src/dataset", img_row=4, img_col=4):
    '''
    Load mnist dataset
    '''
    from array import array as pyarray
    download_mnist(path)
    if dataset == "training_data":
        fname_img = os.path.join(path, "train-images.idx3-ubyte").replace("\\", "/")
        fname_label = os.path.join(path, "train-labels.idx1-ubyte").replace("\\", "/")
    elif dataset == "testing_data":
        fname_img = os.path.join(path, "t10k-images.idx3-ubyte").replace("\\", "/")
        fname_label = os.path.join(path, "t10k-labels.idx1-ubyte").replace("\\", "/")
    else:
        raise ValueError("dataset must be 'testing_data' or 'training_data'")
    
    flbl = open(fname_label, 'rb')
    _, size = struct.unpack(">II", flbl.read(8))

    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)
    images = np.zeros((N, rows, cols))
    labels = np.zeros((N, 1), dtype=int)
    new_images = np.zeros((N, img_row, img_col))
    for i in range(len(ind)):
        images[i] = np.asarray(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols]).reshape((rows, cols))
        new_images[i] = cv2.resize(images[i], (img_row, img_col), interpolation=cv2.INTER_AREA)
        labels[i] = lbl[ind[i]]
    return new_images, labels


if __name__ == '__main__':
    """
    asdas asdasdasdas
    """
    
    images, labels = load_mnist()
    print(images.shape)
    images, labels = load_mnist(digits=[2,3,6])
    print(images.shape)
    print(labels)
    images, labels = load_mnist_5_5()
    print(images.shape)
    images, labels = load_mnist_resize(img_row=12, img_col=12)
    print(images.shape)

