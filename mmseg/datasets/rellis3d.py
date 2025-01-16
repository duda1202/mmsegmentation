# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class Rellis3DDataset(BaseSegDataset):
    """Rellis-3D dataset.

    This dataset uses `.jpg` for images and `.png` for segmentation maps.
    """

    METAINFO = dict(
        classes=(
            'Void', 'Grass', 'Tree', 'Pole', 'Dirt', 'Road', 'Water',
            'Sky', 'Vehicle', 'Building', 'Fence', 'Bush', 'Rock', 'Concrete'
        ),
        palette=[
            [0, 0, 0],         # Void
            [0, 255, 0],       # Grass
            [100, 100, 100],   # Tree
            [0, 100, 255],     # Pole
            [100, 0, 100],     # Dirt
            [255, 0, 255],     # Road
            [0, 255, 255],     # Water
            [255, 255, 0],     # Sky
            [255, 0, 0],       # Vehicle
            [100, 255, 100],   # Building
            [100, 255, 255],   # Fence
            [255, 100, 0],     # Bush
            [0, 0, 255],       # Rock
            [255, 255, 255]    # Concrete
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
