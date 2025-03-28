#themachinethatgoesping.pingprocessing imports
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

#subpackage imports
from .make_wci import make_wci, make_wci_dual_head, make_wci_stack, make_beam_sample_image

class ImageBuilder:

    def __init__(
        self, 
        pings,
        horizontal_pixels,
        wci_render = 'linear',
        progress = False,
        **kwargs):

        self.pings = pings
        self.default_args = {
            "horizontal_pixels" : horizontal_pixels,
        }
        self.default_args.update(kwargs)
        if wci_render == 'beamsample':
            self.beam_sample_view = True
        else:            
            self.beam_sample_view = False
        self.progress = progress

        # if isinstance(self.pings, dict):
        #     self.dual_head = True
        # else:
        #     self.dual_head = False

    def update_args(self, wci_render = 'linear', **kwargs):
        if wci_render == 'beamsample':
            self.beam_sample_view = True
        else:            
            self.beam_sample_view = False
        self.default_args.update(kwargs)

    def build(self, index, stack = 1, stack_step = 1, **kwargs):

        _kwargs = self.default_args.copy()
        _kwargs.update(kwargs)
                

        if stack > 1:
            max_index = index+stack
            if max_index > len(self.pings):
                max_index = len(self.pings)
            
            stack_pings = self.pings[index:max_index:stack_step]
                        
            return make_wci_stack(
                stack_pings,
                progress=self.progress,
                **_kwargs)

        if self.beam_sample_view:
            return make_beam_sample_image(
                self.pings[index],
                **_kwargs)
                

        return make_wci_dual_head(
            self.pings[index],
            **_kwargs)
