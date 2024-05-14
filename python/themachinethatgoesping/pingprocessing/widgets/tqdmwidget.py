from collections import OrderedDict
from ipywidgets import IntProgress

class TqdmWidget(IntProgress):
    def __init__(self, **kwargs):
        _kwargs = {
            "value" : 0,
            "min" : 0,
            "max" : 10,
            "step" : 1,
            "description" : "Idle",
            "orientation" : "horizontal",
        }
        #_kwargs.update((k, kwargs[k]) for k in _kwargs.keys() & kwargs.keys())
        _kwargs.update(kwargs)
        super().__init__(**_kwargs)
        
    def __call__(self, list_like, **kwargs):
        self.list_like = list_like
        self.list_iter = iter(list_like)
        self.index = 0
        self.total = len(list_like)

        if 'desc' in kwargs:
            self.description = kwargs['desc']

        #IntProgress values
        self.max = self.total
        self.value = 0
        
        return self
        
    def __iter__(self):
        return self
    
    def __next__(self):
        self.index += 1
        self.value = self.index
        return next(self.list_iter)
    
    def __len__(self):
        return self.total
    
    def update(self):
        self.index += 1
        next(self.list_iter)
        self.value = self.index
        
    def close(self):
        pass