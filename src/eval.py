import time
import datetime
import pyvqnet
from model.model import Model
from dataloader.mnist import *
from pyvqnet import DEV_GPU_0
from pyvqnet.data.data import data_generator
from pyvqnet.nn.loss import CategoricalCrossEntropy

is_gpu = False

def label_preprocessing(dataset, digits):
    dataset = np.where(dataset == 3, 0, dataset)
    dataset = np.where(dataset == 6, 1, dataset)
    dataset = np.eye(len(digits))[dataset].reshape(-1, len(digits)).astype(np.float32)
    return dataset

def eval():
    epoch_start_time = time.time()
    image_size = 5
    digits = [3,6]
    test_data_size = -1     # -1 means all

    x_test, y_test = load_mnist_resize('testing_data', digits=digits, img_col=image_size, img_row=image_size)
    print('MNIST dataset load successful')
    print('-------------------------------------------------------')
    print(f'Total test image-label pair: {len(x_test)}')
    print(f'Single image shape: {x_test[0].shape}')
    print('-------------------------------------------------------')

    y_test = label_preprocessing(y_test, digits)

    if test_data_size != -1:
        x_test = x_test[:test_data_size]
        y_test = y_test[:test_data_size]

        print('Pre-processing completed')
        print(f'number of test image-label pair: {len(x_test)}')
        print('-------------------------------------------------------')

    model = Model(image_size, 1)
    model_para = pyvqnet.utils.storage.load_parameters('best_model.model')
    model.load_state_dict(model_para)

    if is_gpu:
        model.toGPU(DEV_GPU_0)

    loss = CategoricalCrossEntropy()

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
    elapsed = round(time.time() - epoch_start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Val Finished. Total elapsed time (h:m:s): {}".format(elapsed))

if __name__ == '__main__':
    eval()