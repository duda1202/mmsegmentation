# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class GooseDataset(BaseSegDataset):
    """GOOSE dataset.

    This dataset uses `.jpg` for images and `.png` for segmentation maps.
    """

    METAINFO = dict(
        classes = (
            'undefined', 'traffic_cone', 'snow', 'cobble', 'obstacle', 'leaves',
            'street_light', 'bikeway', 'ego_vehicle', 'pedestrian_crossing',
            'road_block', 'road_marking', 'car', 'bicycle', 'person', 'bus',
            'forest', 'bush', 'moss', 'traffic_light', 'motorcycle', 'sidewalk',
            'curb', 'asphalt', 'gravel', 'boom_barrier', 'rail_track', 'tree_crown',
            'tree_trunk', 'debris', 'crops', 'soil', 'rider', 'animal', 'truck',
            'on_rails', 'caravan', 'trailer', 'building', 'wall', 'rock', 'fence',
            'guard_rail', 'bridge', 'tunnel', 'pole', 'traffic_sign', 'misc_sign',
            'barrier_tape', 'kick_scooter', 'low_grass', 'high_grass',
            'scenery_vegetation', 'sky', 'water', 'wire', 'outlier',
            'heavy_machinery', 'container', 'hedge', 'barrel', 'pipe', 'tree_root',
            'military_vehicle'
        ),
        palette = [
            [0, 0, 0], [0, 255, 255], [160, 87, 209], [255, 52, 255], [70, 74, 255],
            [65, 137, 0], [166, 111, 0], [89, 0, 163], [229, 219, 255], [0, 73, 122],
            [166, 0, 0], [172, 255, 99], [98, 151, 183], [67, 77, 0], [255, 176, 143],
            [135, 125, 153], [7, 0, 90], [147, 150, 128], [189, 168, 180], [0, 68, 27],
            [1, 198, 79], [255, 93, 59], [83, 59, 74], [128, 47, 255], [90, 97, 97],
            [45, 54, 52], [0, 121, 107], [160, 194, 0], [146, 170, 255], [76, 111, 136],
            [237, 134, 0], [0, 97, 209], [255, 239, 221], [53, 0, 0], [75, 79, 123],
            [153, 194, 161], [24, 0, 48], [216, 166, 10], [73, 51, 1], [111, 132, 0],
            [1, 33, 55], [0, 181, 255], [237, 255, 194], [191, 121, 160],
            [68, 7, 204], [178, 185, 192], [153, 255, 194], [9, 30, 0], [89, 196, 190],
            [98, 0, 111], [102, 189, 12], [255, 195, 238], [117, 109, 69],
            [104, 123, 183], [161, 135, 122], [0, 140, 255], [102, 141, 120],
            [159, 208, 250], [154, 138, 255], [23, 211, 232], [0, 208, 208],
            [0, 0, 221], [132, 164, 196], [64, 64, 64]
        ]
    )

    def __init__(self,
                 img_suffix='_windshield_vis.png',
                 seg_map_suffix='_instanceids.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,  # Adjust if necessary
            **kwargs)
