import numpy as np
from scipy.io import loadmat
from ..digits_five import base_dir, join

def load_mnistm(scale=True, usps=False, all_use=False):
    mnistm_data = loadmat(join(base_dir, 'mnistm_with_label.mat'))

    train_data = mnistm_data['train'].transpose(0, 3, 1, 2).astype(np.float32)
    test_data =  mnistm_data['test'].transpose(0, 3, 1, 2).astype(np.float32)
    train_label = np.nonzero(mnistm_data["label_train"])[1]
    test_label = np.nonzero(mnistm_data["label_test"])[1]

    
    if not all_use:
        inds = np.random.permutation(train_data.shape[0])[:25000]
        train_data = train_data[inds]
        train_label = train_label[inds]
        test_data = test_data[:9000]
        test_label = test_label[:9000]
    print(f'mnistm shapes: train X->{train_data.shape}; train Y->{train_label.shape}; test X->{test_data.shape}; train Y->{test_label.shape}')
    return train_data, train_label, test_data, test_label
