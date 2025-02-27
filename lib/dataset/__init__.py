#!/usr/bin/env python

# from .aflw import AFLW
from .cofw import COFW
from .face300w import Face300W
from .wflw import WFLW
from .aflw import AFLW

# __all__ = ['AFLW', 'COFW', 'Face300W', 'WFLW', 'get_dataset']
__all__ = [ 'COFW', 'Face300W', 'WFLW', 'AFLW', 'get_dataset']


def get_dataset(config):
    if config.DATASET.DATASET == 'COFW':
        return COFW
    if config.DATASET.DATASET == 'AFLW':
        return AFLW
    # elif config.DATASET.DATASET == 'COFW':
    #     return COFW
    elif config.DATASET.DATASET == '300W':
        return Face300W
    elif config.DATASET.DATASET == 'WFLW':
        return WFLW
    else:
        raise NotImplemented()
