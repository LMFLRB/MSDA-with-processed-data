import numpy as np
from scipy.io import loadmat
from ..digits_five import base_dir, join

def load_mnist(scale=False, usps=False, all_use=False):
    mnist_data = loadmat(join(base_dir, 'mnist_data.mat'))
    if scale:
        train_data = np.array(mnist_data['train_32'],np.float32)[...,np.newaxis].repeat(3, -1).transpose(0, 3, 1, 2)
        test_data  = np.array(mnist_data['test_32'],np.float32)[...,np.newaxis].repeat(3, -1).transpose(0, 3, 1, 2)
        train_label= np.nonzero(mnist_data["label_train"])[1]
        test_label = np.nonzero(mnist_data['label_test'])[1]
        
    else:
        train_data = np.array(mnist_data['train_28'],np.float32).repeat(3, -1).transpose((0, 3, 1, 2))
        test_data  =  np.array(mnist_data['test_28'],np.float32).repeat(3, -1).transpose((0, 3, 1, 2))
        train_label= np.nonzero(mnist_data["label_train"])[1]
        test_label = np.nonzero(mnist_data['label_test'])[1]
    if not all_use:
        inds = np.random.permutation(train_data.shape[0])[:25000]
        train_data = train_data[inds]
        train_label = train_label[inds]
    print(f'mnist shapes: train X->{train_data.shape}; train Y->{train_label.shape}; test X->{test_data.shape}; train Y->{test_label.shape}')

    return train_data, train_label, test_data , test_label
