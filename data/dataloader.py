import numpy as np

class CaptionsDataLoader():
    def __init__(self, dataset, batch_size, shuffle=True, seed=0):
        self.dataset = dataset
        dataset_indices = list(range(len(dataset)))
        if batch_size == -1:
            batch_size = len(dataset)
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(dataset_indices)
        self.batch_indices = [dataset_indices[batch_size*i:batch_size*(i+1)] for i in range(len(dataset_indices)//batch_size)]
        self.batch_idx = -1

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.batch_indices)

    def __next__(self):
        self.batch_idx+=1
        if self.batch_idx >= len(self.batch_indices):
            self.batch_idx = -1
            raise StopIteration
        else:
            batch_inds = self.batch_indices[self.batch_idx]
            no_returns = len(self.dataset[0])
            batch = [[self.dataset[i][k] for i in batch_inds] for k in range(no_returns)]
            return batch
