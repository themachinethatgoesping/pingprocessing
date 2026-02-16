#themachinethatgoesping.pingprocessing imports
from themachinethatgoesping.pingprocessing.core.progress import get_progress_iterator

#subpackage imports
from .make_wci import make_wci, make_wci_dual_head, make_wci_stack, make_beam_sample_image, downsample_wci

class ImageBuilder:

    def __init__(
        self, 
        pings,
        horizontal_pixels,
        wci_render = 'linear',
        progress = False,
        oversampling = 1,
        oversampling_mode = 'linear_mean',
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
        self.oversampling = max(1, int(oversampling))
        self.oversampling_mode = oversampling_mode

    def update_args(self, wci_render = 'linear', oversampling = None, oversampling_mode = None, **kwargs):
        if wci_render == 'beamsample':
            self.beam_sample_view = True
        else:            
            self.beam_sample_view = False
        if oversampling is not None:
            self.oversampling = max(1, int(oversampling))
        if oversampling_mode is not None:
            self.oversampling_mode = oversampling_mode
        self.default_args.update(kwargs)

    def build(self, index, stack = 1, stack_step = 1, **kwargs):

        _kwargs = self.default_args.copy()
        _kwargs.update(kwargs)
        
        # Apply oversampling: multiply horizontal_pixels
        effective_oversampling = self.oversampling
        if effective_oversampling > 1 and not self.beam_sample_view:
            _kwargs["horizontal_pixels"] = _kwargs["horizontal_pixels"] * effective_oversampling

        if stack > 1:
            max_index = index+stack
            if max_index > len(self.pings):
                max_index = len(self.pings)
            
            stack_pings = self.pings[index:max_index:stack_step]
                        
            wci, extent = make_wci_stack(
                stack_pings,
                progress=self.progress,
                **_kwargs)
        elif self.beam_sample_view:
            wci, extent = make_beam_sample_image(
                self.pings[index],
                **_kwargs)
        else:
            wci, extent = make_wci_dual_head(
                self.pings[index],
                **_kwargs)
        
        # Downsample if oversampling was applied
        if effective_oversampling > 1 and not self.beam_sample_view:
            wci, extent = downsample_wci(wci, extent, effective_oversampling, mode=self.oversampling_mode)
        
        return wci, extent
