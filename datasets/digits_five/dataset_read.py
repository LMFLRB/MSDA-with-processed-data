from .svhn import load_svhn
from .mnist import load_mnist
from .mnist_m import load_mnistm
from .usps import load_usps
from .gtsrb import load_gtsrb
from .synthetic import load_synthetic
from .syntraffic import load_syntraffic

def return_DigitsFive(domain, scale=False, usps=False, all_use=False, **kwrags):
    if domain in ['svhn', 'mnist', 'mnistm', 'usps', 'syntraffic', 'gtsrb', 'synthetic']:
        train_data, train_label, test_data, test_label = globals()[f'load_{domain}'](scale, usps, all_use)
        # train_label, test_label = train_label, test_label
        return dict(train=dict(imgs=train_data, labels=train_label), 
                    test=dict(imgs=test_data, labels=test_label))
    else:
        Warning(f"no module for data {domain}")