from copy import copy

import numpy as np

# internal pingprocessing.watercolumn imports
from themachinethatgoesping.pingprocessing.core.asserts import assert_length

# submodules
from .echolayer import EchoLayer

class LayerGenerator:
    """Generate processing layers on a base echogram and copy them to a second echogram.
    
    Designed for the new EchogramBuilder (backend-based).
    Layers are created in range (or depth) space on ``echogram_base`` and then
    converted to depth coordinates on ``echogram_second``.
    """

    def __init__(self, echogram_base, echogram_second, cut_in_range=True,
                 minslant_relative=0.95, minslant_absolute=-0.5):
        echogram_base.clear_layers()
        echogram_second.clear_layers()

        if echogram_base.get_y_axis_name() != 'Range (m)':
            echogram_base.set_y_axis_range()
        
        self.valid = EchoLayer.from_ping_param_offsets_relative(echogram_base, 'minslant', None, minslant_relative)
        self.valid.combine(EchoLayer.from_ping_param_offsets_absolute(echogram_base, 'minslant', None, minslant_absolute))
        self.valid.combine(EchoLayer.from_ping_param_offsets_relative(echogram_base, 'bubbles', 1, None))

        self.cut_in_range = cut_in_range
        self.echogram_base = echogram_base
        self.echogram_second = echogram_second

    def add_layer(self, layer_range, layer_size=1):
        layer_name = self.__make_layer__(layer_range, layer_size)
        self.__copy_layer__(self.echogram_second, layer_name)

    def __make_layer__(self, layer_range, layer_size=1):
        if self.cut_in_range:
            if self.echogram_base.get_y_axis_name() != 'Range (m)':
                self.echogram_base.set_y_axis_range()
        else:
            if self.echogram_base.get_y_axis_name() != 'Depth (m)':
                self.echogram_base.set_y_axis_depth()
            
        layer_name = f'{layer_range}m'
        self.echogram_base.layers[layer_name] = copy(self.valid)
        self.echogram_base.add_layer_from_static_layer(
            layer_name, layer_range - layer_size * 0.5, layer_range + layer_size * 0.5)
        return layer_name

    def __copy_layer__(self, echogram2, layer_name):
        """Copy a layer from echogram_base to echogram2, converting to depth coordinates.
        
        Uses the base echogram's affine transforms to convert sample indices
        to depth values, then adds the layer to echogram2 in depth space.
        """
        if echogram2.get_y_axis_name() != 'Depth (m)':
            echogram2.set_y_axis_depth()

        if layer_name not in self.echogram_base.layers:
            raise RuntimeError(f'__copy_layer__: layer "{layer_name}" not found')
        
        layer = self.echogram_base.layers[layer_name]
        cs = self.echogram_base._coord_system
        
        # Convert sample indices → depth using affine: depth = a + b * sample_index
        affine = cs._affine_sample_to_depth
        if affine is None:
            raise RuntimeError(
                '__copy_layer__: depth extents not set on echogram_base. '
                'Call set_depth_extent() first.'
            )
        a, b = affine
        
        vec_y0 = a + b * layer.i0.astype(np.float64)
        vec_y1 = a + b * layer.i1.astype(np.float64)
        
        # Use ping times as x-values (works regardless of current x-axis setting)
        vec_x = self.echogram_base._backend.ping_times
        
        echogram2.add_layer(layer_name, vec_x, vec_y0, vec_y1)