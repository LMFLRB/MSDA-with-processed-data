import numpy as np
from scipy.io import loadmat
from ..digits_five import base_dir

def load_usps(scale=False, usps=False, all_use=False):
    dataset  = loadmat(base_dir + '/usps_28x28.mat')['dataset']
    train_data = (dataset[0][0]* 255).astype(np.uint8).repeat(3, 1)
    test_data = (dataset[1][0]* 255).astype(np.uint8).repeat(3, 1)
    train_label = dataset[0][1][:,0]
    test_label = dataset[1][1][:,0]

    inds = np.random.permutation(train_data.shape[0])
    train_data = train_data[inds]
    train_label = train_label[inds]
    
    # train_data = np.concatenate([train_data, train_data, train_data, train_data], 0)
    # train_label = np.concatenate([train_label, train_label, train_label, train_label], 0)
    
    print(f'usps shapes: train X->{train_data.shape}; train Y->{train_label.shape}; test X->{test_data.shape}; train Y->{test_label.shape}')

    return train_data, train_label, test_data, test_label
