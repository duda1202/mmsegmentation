# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class ENTFACDataset(BaseSegDataset):
    """GOOSE dataset.

    This dataset uses `.jpg` for images and `.png` for segmentation maps.
    """

    METAINFO = dict(
        classes=(
            'Tree',      # Includes trees, tree trunks, and related structures
            'Bush',      # Includes bushes, hedges, and similar vegetation
            'Grass',     # Includes grasslands, moss, and low/high grass
            'Sky',       # The sky class
            'Water',     # Water bodies
            'Vegetation', # Other vegetation like crops, forests, and scenery vegetation
            'Others'
        ),
        palette=(
            [34, 139, 34],  # Green for Tree
            [85, 107, 47],  # Olive for Bush
            [124, 252, 0],  # LawnGreen for Grass
            [135, 206, 235],# SkyBlue for Sky
            [0, 0, 255],    # Blue for Water
            [50, 205, 50],   # LimeGreen for Vegetation
            [0, 0, 0]       # Black for Others
        )
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,  # Adjust if necessary
            **kwargs)
