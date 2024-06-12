# Model Training

此处以QCL为例进行在MNIST数据集上的二分类模型训练，

首先是一些头文件引入和超参数的定义

```python
from pyqpanda import *
from pyvqnet.optim.adam import Adam
from pyvqnet.nn.loss import CategoricalCrossEntropy
from pyvqnet.data.data import data_generator
import datetime
import numpy as np
import cv2
from model.model import Model

from dataloader.mnist import *
from utils import create_result_file

def run():
		batch_size = 20
		epoch = 10
		image_size = 4
		digits = [3,6]
		train_data_size = -1    # -1 means all
		test_data_size = -1     # -1 means all
    ...
```

这里我们将mnist压缩到像素为4*4的图像进行训练；batch_size和epoch设置为20和10；选取标签为3和6的数据；dataset不进行截取。

下面是数据集加载

```python
def run():
  	...
    x_train, y_train = load_mnist_resize('training_data', digits=digits, img_col=image_size, img_row=image_size)
    x_test, y_test = load_mnist_resize('testing_data', digits=digits, img_col=image_size, img_row=image_size)
    print('MNIST dataset load successful')
    print('-------------------------------------------------------')
    print(f'Total train image-label pair: {len(x_train)}')
    print(f'Total test image-label pair: {len(x_test)}')
    print(f'Single image shape: {x_train[0].shape}')
    print('-------------------------------------------------------')

    y_train = label_preprocessing(y_train, digits)
    y_test = label_preprocessing(y_test, digits)

    # x_train = x_train / 255
    # x_train = x_test / 255

    if train_data_size != -1 or test_data_size != -1:
        x_train = x_train[:train_data_size]
        y_train = y_train[:train_data_size]
        x_test = x_test[:test_data_size]
        y_test = y_test[:test_data_size]

        print('Pre-processing completed')
        print(f'number of train image-label pair: {len(x_train)}')
        print(f'number of test image-label pair: {len(x_test)}')
        print('-------------------------------------------------------')
    ...
```

由于我们截取标签为3和6的图像进行二分类训练，[3]或者[6]不符合二分类标签要求，因此需要将其处理成[0., 1.]的ont-hot形式。

```python
def label_preprocessing(dataset, digits):
    dataset = np.where(dataset == 3, 0, dataset)
    dataset = np.where(dataset == 6, 1, dataset)
    dataset = np.eye(len(digits))[dataset].reshape(-1, len(digits)).astype(np.float32)
    return dataset
```

并不难，直接进行替换即可。到此数据集已经准备好了，可以开始模型的训练过程。

model的声明和优化器与经典神经网络是一样的

```python
def run():
		...
		model = Model(image_size, 1)
    optimizer = Adam(model.parameters(), lr = 3e-3)
    loss = CategoricalCrossEntropy()
    model.train()
    ...
```

这里我们选用Adam优化器，学习率设置为0.003，损失函数为交叉熵损失函数

训练过程如下：

```python
def run():
	  ...
		start_time = time.time()
    for i in range(epoch):
        epoch_start_time = time.time()
        count = 0
        sum_loss = 0
        accuary = 0
        t = 0
        for x, y in data_generator(x_train, y_train, batch_size, shuffle=True):
            iter_time = time.time()
            x = x.reshape(batch_size, -1)
            optimizer.zero_grad()
            result = model(x)
            loss_b = loss(y, result)
            loss_b.backward()
            optimizer._step()
            sum_loss += loss_b.item()
            count += batch_size
    
            np_output = np.array(result.data, copy=False)
            mask = np_output.argmax(1) == y.argmax(1)
            accuary += np.sum(mask)

            print(f'epoch: {i + 1}, lr: {optimizer.lr} batch: {count}: {len(x_train)}, loss: {loss_b.item(): .7f}', end=", ")
            print(f'correct: {accuary / count: .3f}')
            print('iteration time: {:.3f}s'.format(time.time() - iter_time))


        print(f"epoch: {i + 1} #### loss: {sum_loss * batch_size / count} #### accuary: {accuary / count: 01.2f}")
        result_file.write(f'{i + 1} | {sum_loss * batch_size / count: .17f} | {accuary / count: 01.4f} | ')
        elapsed = round(time.time() - epoch_start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Epoch Finished. Total elapsed time (h:m:s): {}".format(elapsed))

        epoch_start_time = time.time()
        model.eval()
        count = 0
        test_batch_size = 1
        accuary = 0
        sum_loss = 0
        for x, y in data_generator(x_test, y_test, test_batch_size, shuffle=False):
            x = x.reshape(test_batch_size, -1)
            test_result = model(x)
            test_loss = loss(y, test_result)
            sum_loss += test_loss.item()
            count += test_batch_size

            np_output = np.array(test_result.data, copy=False)
            mask = np_output.argmax(1) == y.argmax(1)
            accuary += np.sum(mask)
        
        print(f"eval:--------------> loss: {sum_loss / count: .17f} #### accuary: {accuary / count: 01.2f}")
        result_file.write(f'{sum_loss / count: .17f} | {accuary / count: 01.4f}\n')
        elapsed = round(time.time() - epoch_start_time)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Val Finished. Total elapsed time (h:m:s): {}".format(elapsed))

        result_file.flush()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Train finished. Total elapsed time (h:m:s): {}".format(elapsed))
```

可以看到整个训练过程与经典神经网络训练是一样的。

以下是一次训练的结果：

```python
epoch # train_loss ### train_accuracy # test_loss ### eval_accuracy
1 |  0.20050946168601513 |  0.7410 | 0.1348683688212186 |  0.9270
2 |  0.10102434538304805 |  0.9716 | 0.10213165939040482 |  0.9660
3 |  0.08536552682518959 |  0.9766 | 0.0932164337025024 |  0.9670
4 |  0.07922543372958898 |  0.9770 | 0.08875624094204977 |  0.9650
5 |  0.07640370458364487 |  0.9794 | 0.08705172596639023 |  0.9660
6 |  0.07546331323683261 |  0.9800 | 0.08569387398753316 |  0.9700
7 |  0.07425988867878913 |  0.9802 | 0.0833789216442965 |  0.9720
8 |  0.07279767476022243 |  0.9802 | 0.08255305851553567 |  0.9730
9 |  0.07151625573635101 |  0.9822 | 0.08097505645640195 |  0.9730
10 |  0.07073687914758921 |  0.9824 | 0.07974313347763382 |  0.9760
Train finished. Total elapsed time (h:m:s): 0:10:58
```

