import numpy as np
from scipy.io import loadmat
from ..digits_five import base_dir, join


def load_synthetic(scale=True, usps=False, all_use=False):
    syn_data = loadmat(join(base_dir, 'syn_number.mat'))
    train_data = syn_data['train_data'].transpose(0, 3, 1, 2).astype(np.float32)
    test_data =  syn_data['test_data'].transpose(0, 3, 1, 2).astype(np.float32)
    train_label = syn_data['train_label'][:,0]
    test_label = syn_data['test_label'][:,0]

    print(f'synthetic shapes: train X->{train_data.shape}; train Y->{train_label.shape}; test X->{test_data.shape}; train Y->{test_label.shape}')
    return train_data, train_label, test_data, test_label
