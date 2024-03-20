from torch.utils.data import DataLoader
from builtins import object

class UnalignedDataLoader(object):
    def __init__(self, sources, batch_size=8, max_size=float("inf"), drop_last=True):
        kwargs = dict(shuffle=True, num_workers=0, drop_last=drop_last)
        self.dataloaders = [DataLoader(source, batch_size=batch_size, **kwargs) for source in sources]
        self.n_loaders=len(self.dataloaders)
        self.stops = [False]*self.n_loaders
        self.max_size = max_size        
        self.feature_shapes=[domain.feature_shape if hasattr(domain, 'feature_shape') else domain.dataset.feature_shape for domain in sources]   

    def __iter__(self):
        self.stops = [False]*self.n_loaders
        self.dataiters = [iter(dataloader) for dataloader in self.dataloaders]
        self.iter = 0
        return self

    def __next__(self):
        if len(self.dataloaders)>1:
            As, A_paths = [None]*self.n_loaders, [None]*self.n_loaders
            for num, (dataloader, dataiter) in enumerate(zip(self.dataloaders, self.dataiters)):
                As[num], A_paths[num] = self.align_next(num, dataloader, dataiter)
        else:      
            As, A_paths = self.align_next(0, self.dataloaders[0], self.dataiters[0])
        if sum(self.stops)==len(self.stops) or self.iter > self.max_size-1:
            self.stops = [False]*self.n_loaders
            raise StopIteration()
        else:
            self.iter += 1
            return As, A_paths
        
    def __len__(self):
        return min(max([len(dataloader) for dataloader in self.dataloaders]), self.max_size)
        
    def align_next(self, num, dataloader, dataiter):
        A, A_path = None, None
        try:
            A, A_path = next(dataiter)
        except StopIteration:
            if A is None or A_path is None:
                self.stops[num] = True
                self.dataiters[num] = iter(dataloader)
                A, A_path = next(self.dataiters[num])  
        return A, A_path