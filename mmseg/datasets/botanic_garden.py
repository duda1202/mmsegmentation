#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-*- coding: utf-8 -*-

Maintainer: Duda Andrada <duda.andrada@isr.uc.pt>
Author/s: 
Written: January 2025
License: This code is licensed under the MIT License.

Program: BotanicGarden Dataset Loader
Purpose: Define dataset loader for the BotanicGarden semantic segmentation dataset.
"""

from .basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS
@DATASETS.register_module()
class BotanicGardenDataset(BaseSegDataset):
    METAINFO = {
        'classes': [
            'tree trunk', 'tree', 'bush', 'grassland', 'road', 'sky', 'grass',
            'others', 'water', 'facility', 'chair', 'flower bed', 'water plants',
            'bridge', 'building', 'fencing', 'sign', 'dustbin', 'person',
            'lamp', 'swing', 'bike', 'column', 'door', 'cover', 'rider', 'trike'
        ],
        'palette': [
            [128, 64, 128],  # tree trunk
            [244, 35, 232],  # tree
            [70, 70, 70],    # bush
            [107, 142, 35],  # grassland
            [153, 153, 153], # road
            [220, 20, 60],   # sky
            [119, 11, 32],   # grass
            [0, 0, 142],     # others
            [0, 0, 230],     # water
            [106, 0, 228],   # facility
            [0, 60, 100],    # chair
            [0, 80, 100],    # flower bed
            [0, 0, 70],      # water plants
            [0, 100, 100],   # bridge
            [0, 200, 100],   # building
            [50, 0, 100],    # fencing
            [60, 0, 100],    # sign
            [170, 0, 100],   # dustbin
            [192, 128, 128], # person
            [190, 153, 153], # lamp
            [180, 165, 180], # swing
            [0, 100, 200],   # bike
            [100, 100, 100], # column
            [200, 200, 100], # door
            [150, 150, 150], # cover
            [250, 0, 30],    # rider
            [220, 220, 0]    # trike
        ]
    }

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.tif',  # Adjust as needed
            seg_map_suffix='.png',  # Adjust as needed
            reduce_zero_label=False,
            **kwargs
        )
