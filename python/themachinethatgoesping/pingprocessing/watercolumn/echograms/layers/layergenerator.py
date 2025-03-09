from copy import copy


# internal pingprocessing.watercolumn imports
from themachinethatgoesping.pingprocessing.core.asserts import assert_length

# submodules
from .echolayer import EchoLayer

class LayerGenerator:
    def __init__(self, echogram_base, echogram_second, cut_in_range=True, minslant_relative=0.95, minslant_absolute=-0.5):
        echogram_base.clear_layers()
        echogram_second.clear_layers()

        if echogram_base.y_axis_name != 'Range (m)':
            echogram_base.set_y_axis_range()
        
        self.valid = EchoLayer.from_ping_param_offsets_relative(echogram_base, 'minslant', None, minslant_relative)
        self.valid.combine(EchoLayer.from_ping_param_offsets_absolute(echogram_base, 'minslant', None, minslant_absolute))
        self.valid.combine(EchoLayer.from_ping_param_offsets_relative(echogram_base, 'bubbles', 1, None))

        self.cut_in_range = cut_in_range
        self.echogram_base = echogram_base
        self.echogram_second = echogram_second

    def add_layer(self, layer_range, layer_size = 1):
        layer_name = self.__make_layer__(layer_range, layer_size)
        self.__copy_layer__(self.echogram_second, layer_name)

    def __make_layer__(self, layer_range, layer_size=1):
        if self.cut_in_range:
            if self.echogram_base.y_axis_name != 'Range (m)':
                self.echogram_base.set_y_axis_range()
        else:
            if self.echogram_base.y_axis_name != 'Depth (m)':
                self.echogram_base.set_y_axis_depth()
            
        layer_name = f'{layer_range}m'
        self.echogram_base.layers[layer_name] = copy(self.valid)
        self.echogram_base.add_layer_from_static_layer(layer_name,layer_range-layer_size*0.5, layer_range + layer_size*0.5)
        return layer_name

    def __copy_layer__(self, echogram2, layer_name):
        if echogram2.y_axis_name != 'Depth (m)':
            echogram2.set_y_axis_depth()

        if not  layer_name in self.echogram_base.layers.keys():
            raise RuntimeError('__copy_layer__: Aaaah')
        
        layer = self.echogram_base.layers[layer_name]
        
        vec_x, vec_y0, vec_y1 = self.echogram_base.vec_x_val,[], []
        for i,interpolator in enumerate(self.echogram_base.y_indice_to_depth_interpolator):
            if interpolator is not None:
                vec_y0.append(interpolator(layer.i0[i]))
                vec_y1.append(interpolator(layer.i1[i]))
            else:
                vec_y0.append(0)
                vec_y1.append(0)
        
        echogram2.add_layer(layer_name, vec_x, vec_y0, vec_y1)