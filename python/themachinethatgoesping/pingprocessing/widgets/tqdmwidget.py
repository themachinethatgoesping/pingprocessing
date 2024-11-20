import ipywidgets
from tqdm.notebook import tqdm

class TqdmWidget(ipywidgets.HBox):
    def __init__(self, **kwargs):
        self.kwargs = {
            "desc" : "Idle",
        }
        #_kwargs.update((k, kwargs[k]) for k in _kwargs.keys() & kwargs.keys())
        self.kwargs.update(kwargs)
        super().__init__()
        
    def set_description(self, desc):
        self.kwargs["desc"] = desc

    def __call__(self, list_like, **kwargs):
        self.list_like = list_like

        self.kwargs.update(kwargs)
        
        self.progress = tqdm(self.list_like, display=False, **self.kwargs)
        self.children=[self.progress.container]
        self.progress_iter = iter(self.progress)
        
        return self
        
    def __iter__(self):
        return self
    
    def __next__(self):
        return next(self.progress_iter)
    
    def __len__(self):
        return self.progress.total
    
    def update(self):
        next(self.progress_iter)
        #self.progress.tick()
        
    def close(self):
        self.progress.close()