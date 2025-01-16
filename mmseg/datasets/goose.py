# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class GooseDataset(BaseSegDataset):
    """GOOSE dataset.

    This dataset uses `.jpg` for images and `.png` for segmentation maps.
    """

    METAINFO = dict(
        classes=(
            'animal', 'bridge', 'building', 'container', 'debris', 'fence',
            'guard_rail', 'tunnel', 'wall', 'wire', 'person', 'rider',
            'obstacle', 'pole', 'street_light', 'rock', 'barrel', 'pipe',
            'bikeway', 'curb', 'pedestrian_crossing', 'rail_track',
            'road_marking', 'sidewalk', 'barrier_tape', 'misc_sign',
            'traffic_cone', 'traffic_light', 'traffic_sign', 'road_block',
            'boom_barrier', 'sky', 'asphalt', 'cobble', 'gravel', 'soil',
            'snow', 'bush', 'crops', 'forest', 'hedge', 'high_grass', 'leaves',
            'low_grass', 'moss', 'scenery_vegetation', 'tree_crown',
            'tree_trunk', 'tree_root', 'bicycle', 'bus', 'car', 'caravan',
            'heavy_machinery', 'kick_scooter', 'motorcycle', 'on_rails',
            'trailer', 'truck', 'ego_vehicle', 'outlier', 'undefined', 'water'
        ),
        palette=[
            [64, 128, 64], [192, 0, 128], [0, 128, 192], [0, 128, 64],
            [128, 0, 64], [64, 0, 192], [192, 128, 64], [192, 192, 128],
            [64, 64, 128], [128, 0, 192], [192, 0, 64], [128, 128, 64],
            [0, 0, 192], [192, 0, 192], [64, 192, 128], [64, 64, 0],
            [128, 64, 0], [192, 64, 0], [0, 192, 192], [128, 192, 192],
            [0, 64, 64], [192, 64, 128], [128, 64, 128], [64, 192, 192],
            [0, 192, 64], [128, 192, 64], [192, 192, 64], [64, 0, 64],
            [192, 0, 0], [64, 128, 192], [192, 128, 192], [128, 128, 192],
            [0, 0, 64], [0, 64, 192], [128, 0, 0], [64, 0, 0], [192, 0, 64],
            [0, 192, 0], [64, 192, 0], [128, 192, 0], [192, 192, 0],
            [0, 64, 0], [64, 64, 64], [128, 64, 64], [192, 64, 64],
            [0, 128, 0], [64, 128, 0], [128, 128, 0], [192, 128, 0],
            [0, 0, 128], [64, 0, 128], [128, 0, 128], [192, 0, 128],
            [0, 128, 128], [64, 128, 128], [128, 128, 128], [192, 128, 128],
            [0, 64, 128], [64, 64, 128], [128, 64, 128], [192, 64, 128],
            [0, 192, 128], [64, 192, 128], [128, 192, 128], [192, 192, 128]
        ]
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
