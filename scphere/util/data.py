import numpy as np

MAX_VALUE = 1
LABEL = None

np.random.seed(0)


class DataSet:
    
    def __init__(self, x, y=LABEL, max_value=MAX_VALUE, shuffle=True):
        if y is not None:
            assert x.shape[0] == y.shape[0]
        
        self._num_data = x.shape[0]
        x = x.astype(np.float32)
        x /= max_value
        
        self._x = x
        self._y = y
        self._epoch = 0
        self._index_in_epoch = 0
        
        index = np.arange(self._num_data)
        if shuffle:
            np.random.shuffle(index)
            
        self._index = index
        
        
    @property
    def all_data(self):
        return self._x
    
    @property
    def label(self):
        self._y
        
    @property
    def num_of_data(self):
        return self._num_data
    
    @property
    def complete_epoch(self):
        return self._epoch
    
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        
        if self._index_in_epoch > self._num_data:
            assert batch_size <= self._num_data
            
            self._epoch += 1
            np.random.shuffle(self._index)
            
            start = 0
            self._index_in_epoch = batch_size
            
        end = self._index_in_epoch
        
        index = self._index[start:end]
        if self._y is not None:
            y = self._y[index]
        else:
            y = self._y
            
        return self._x[index], y
    
    
    
def prepare_data(data):
    
    class Data:
        pass
    
    data_set = Data()
    
    assert isinstance(data, tuple), ('Data should be a tuple: %s' % type(data))
              
    for item in data:
        assert isinstance(item, tuple), ('Each element of data should be a tuple: %s' % type(item))

    num_dataset = data.__len__()
    data_set.len = num_dataset

    assert data_set.len <= 3 & data_set.len >= 1, \
        ('Number of datasets: %s' % num_dataset)

    if len(data[0]) == 2:
        data_set.train = DataSet(data[0][0], data[0][1])
    else:
        data_set.train = DataSet(data[0][0])

    if num_dataset > 1:
        if len(data[1]) == 2:
            data_set.validation = DataSet(data[1][0], data[1][1])
        else:
            data_set.validation = DataSet(data[1][0])

    if num_dataset == 3:
        if len(data[2]) == 2:
            data_set.test = DataSet(data[2][0], data[2][1])
        else:
            data_set.test = DataSet(data[2][0])

    return data_set