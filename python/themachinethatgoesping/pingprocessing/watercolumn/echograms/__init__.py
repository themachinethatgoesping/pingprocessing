# folders

# modules
from . import layers
from . import backends

# functions
from .echodata import EchoData
from .echogrambuilder import EchogramBuilder

# backends (convenience exports)
from .backends import EchogramDataBackend, PingDataBackend, ZarrDataBackend

# New refactored classes
from .coordinate_system import EchogramCoordinateSystem
from .echogrambuilder_new import EchogramBuilder as EchogramBuilderNew
from .indexers import EchogramImageRequest


def is_new_builder(echogram) -> bool:
    """Check if an echogram instance is the new EchogramBuilderNew with LayerManager.
    
    Args:
        echogram: An EchogramBuilder or EchogramBuilderNew instance.
        
    Returns:
        True if using EchogramBuilderNew (with LayerManager), False otherwise.
    """
    return hasattr(echogram, '_layer_manager')


def is_refactored_builder(echogram) -> bool:
    """Check if an echogram instance uses the refactored builder with separate coordinate system.
    
    Args:
        echogram: An EchogramBuilder instance.
        
    Returns:
        True if using EchogramBuilderRefactored (with separate coord_system), False otherwise.
    """
    return hasattr(echogram, '_coord_system')


def has_layers(echogram) -> bool:
    """Check if an echogram has any layers (works with both builder types).
    
    Args:
        echogram: An EchogramBuilder or EchogramBuilderNew instance.
        
    Returns:
        True if there are named layers or a main layer.
    """
    if is_new_builder(echogram):
        return echogram._layer_manager.has_layers()
    else:
        return len(echogram.layers.keys())>0 or echogram.main_layer is not None

