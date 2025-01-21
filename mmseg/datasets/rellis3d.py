# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class Rellis3DDataset(BaseSegDataset):
    """Rellis-3D dataset.

    This dataset uses `.tif` for images and `.png` for segmentation maps.
    """

    METAINFO = dict(
        classes=(
            'void', 'dirt', 'grass', 'tree', 'pole', 'water', 'sky', 'vehicle',
            'object', 'asphalt', 'building', 'log', 'person', 'fence',
            'bush', 'concrete', 'barrier', 'puddle', 'mud', 'rubble'
        ),
        palette=[
            [0, 0, 0],          # void
            [108, 64, 20],      # dirt
            [0, 102, 0],        # grass
            [0, 255, 0],        # tree
            [0, 153, 153],      # pole
            [0, 128, 255],      # water
            [0, 0, 255],        # sky
            [255, 255, 0],      # vehicle
            [255, 0, 127],      # object
            [64, 64, 64],       # asphalt
            [255, 0, 0],        # building
            [102, 0, 0],        # log
            [204, 153, 255],    # person
            [102, 0, 204],      # fence
            [255, 153, 204],    # bush
            [170, 170, 170],    # concrete
            [41, 121, 255],     # barrier
            [134, 255, 239],    # puddle
            [99, 66, 34],       # mud
            [110, 22, 138]      # rubble
        ])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs)
