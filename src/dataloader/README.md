# Dataloader

MNIST数据集是一个大型手写数字数据库，包含60000张训练图像和10000张测试图像，通常用于训练各种图像处理系统。所有的手写数字图片的分辨率为**28*28**。

| Dataset        | File name                  | Byte size   |
| -------------- | -------------------------- | ----------- |
| Training image | train-images-idx3-ubyte.gz | 9912422字节 |
| Training label | train-labels-idx1-ubyte.gz | 28881字节   |
| Testing image  | t10k-images-idx3-ubyte.gz  | 1648877字节 |
| Testing label  | t10k-labels-idx1-ubyte.gz  | 4542字节    |

----

下面让我们开始数据预处理

首先导入必要的类库

```python 
import os
import os.path
import struct
import gzip
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

try:
    matplotlib.use('TkAgg')
except: # pylint: disable=bare-except
    print('Can not use matplot TkAgg')
    pass

try:
    import urllib.request
except ImportError:
    raise ImportError('You should use python 3.x')
```

接着是数据集准备

```python
# Official address
#url_base = "http://yann.lecun.com/exdb/mnist/"
# Local address
url_base = 'file:your-mnist-file-address'

key_file = {
		"train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz"
}
```

建议使用本地地址进行加载，只需将地址修改为本地地址即可

下面是数据集下载

```python
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
```

这里无需理解

```python
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
```

加载mnist数据集，总共分三部分。形参有三个，分别是dataset选择、数据类选择、数据集地址。

第一部分，数据集下载及地址准备，这部分包含数据集下载解压全过程，完成后提供可读取的对应dataset文件地址

```python
...
		download_mnist(path)
    if dataset == "training_data":
        fname_img = os.path.join(path, "train-images.idx3-ubyte").replace("\\", "/")
        fname_label = os.path.join(path, "train-labels.idx1-ubyte").replace("\\", "/")
    elif dataset == "testing_data":
        fname_img = os.path.join(path, "t10k-images.idx3-ubyte").replace("\\", "/")
        fname_label = os.path.join(path, "t10k-labels.idx1-ubyte").replace("\\", "/")
    else:
        raise ValueError("dataset must be 'testing_data' or 'training_data'")
...
```

第二部分，数据集读取，这部分为dataset的图像和标签读取，细节无需理解，完成后提供完整的dataset数据

```python
...
    flbl = open(fname_label, 'rb')
    _, size = struct.unpack(">II", flbl.read(8))

    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    _, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
...
```

第三部分，数据集完备，这部分为数据集预处理部分，完成后提供数据集数据

```python
...
		ind = [k for k in range(size) if lbl[k] in digits]	# 生成需求的类别索引
    N = len(ind)																				
    images = np.zeros((N, rows, cols))	# 准备images的空numpy矩阵，用于存储图像
    labels = np.zeros((N, 1), dtype=int)	# 准备labels的空numpy矩阵，用于存储标签
    for i in range(len(ind)):
        images[i] = np.asarray(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols]).reshape((rows, cols))	# 根据ind索引保留数据、标签
        labels[i] = lbl[ind[i]]
...
```

至此，数据集准备完毕，下面来看看生成结果

```python
images, labels = load_mnist()
print(images.shape)
images, labels = load_mnist(digits=[2,3,6])
print(images.shape)
print(labels)

"""
(60000, 28, 28)
(18007, 28, 28)
[[2]
 [3]
 [3]
 ...
 [2]
 [3]
 [6]]
"""
```

其中digits为类别索引，digits=[2,3,6]表示生成的dataset的类别仅包含2、3、6。

此外增加了resize来获取不同尺寸的dataset图像以适应不同的model，仅多了一行cv2的resize()。

```python
def load_mnist_resize(dataset='training_data', digits=np.arange(10), path="./src/dataset", img_row=4, img_col=4):
		...
    images = np.zeros((N, rows, cols))
    labels = np.zeros((N, 1), dtype=int)
    new_images = np.zeros((N, img_row, img_col))
    for i in range(len(ind)):
        images[i] = np.asarray(img[ind[i]*rows*cols : (ind[i]+1)*rows*cols]).reshape((rows, cols))
        new_images[i] = cv2.resize(images[i], (img_row, img_col), interpolation=cv2.INTER_AREA)
        labels[i] = lbl[ind[i]]
    return new_images, labels

if __name__ == '__main__':
		images, labels = load_mnist_resize(img_row=12, img_col=12)

# output: (60000, 12, 12)
```

