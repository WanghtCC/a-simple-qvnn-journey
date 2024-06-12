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

qvc_block = 2   # number of qvc block, >= 1
single_line_weight = 6  # number of qcl weights, >= 3

def label_preprocessing(dataset, digits):
    dataset = np.where(dataset == 3, 0, dataset)
    dataset = np.where(dataset == 6, 1, dataset)
    dataset = np.eye(len(digits))[dataset].reshape(-1, len(digits)).astype(np.float32)
    return dataset

def run():
    batch_size = 20
    epoch = 10
    image_size = 4
    digits = [3,6]
    train_data_size = -1    # -1 means all
    test_data_size = -1     # -1 means all

    x_train, y_train = load_mnist_resize('training_data', digits=digits)
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
    
    result_file = create_result_file("qcl")
    result_file.write(f'train image-label pair: {len(x_train)}\n')
    result_file.write(f'test image-label pair: {len(x_test)}\n')
    result_file.write(f'epoch # train_loss ### train_accuracy # test_loss ### eval_accuracy\n')

    model = Model(image_size, 1)
    optimizer = Adam(model.parameters(), lr = 0.003)
    loss = CategoricalCrossEntropy()
    model.train()

    start_time = time.time()
    for i in range(epoch):
        epoch_start_time = time.time()
        count = 0
        sum_loss = 0
        accuary = 0
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
    result_file.write(f'Train finished. Total elapsed time (h:m:s): {elapsed}\n')
    result_file.close()


if __name__ == '__main__':
    run()
