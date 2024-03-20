from scipy.io import loadmat
import numpy as np
from ..digits_five import base_dir, join

def load_svhn(scale=False, usps=False, all_use=False):
    train = loadmat(join(base_dir, 'svhn_train_32x32.mat'))
    train_data = train['X'].transpose(3, 2, 0, 1).astype(np.float32)
    train_label = train['y'][:,0]-1

    test = loadmat(join(base_dir, 'svhn_test_32x32.mat'))
    test_data = test['X'].transpose(3, 2, 0, 1).astype(np.float32)
    test_label = test['y'][:,0]-1
    if not all_use:
        inds = np.random.permutation(train_data.shape[0])[:25000]
        train_data = train_data[inds]
        train_label = train_label[inds]
        test_data = test_data[:9000]
        test_label = test_label[:9000]

    print(f'svhn shapes: train X->{train_data.shape}; train Y->{train_label.shape}; test X->{test_data.shape}; train Y->{test_label.shape}')

    return train_data, train_label, test_data, test_label
