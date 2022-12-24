#!/usr/bin/env python
# coding: utf-8

# Refer to:
# https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/

from skimage.metrics import structural_similarity as ssim
import os
import numpy as np
import cv2
import warnings

GT_DIR = 'stereo_GT'
RES_DIR = 'output'

gt_imgs = ['AdirondackGT.png', 'JadeplantGT.png', 'MotorcycleGT.png', 'PianoGT.png',
            'PipesGT.png', 'PlayroomGT.png', 'PlaytableGT.png', 'RecycleGT.png',
            'ShelvesGT.png', 'TeddyGT.png', 'VintageGT.png']

res_w5 = ['Adirondack_ssd_dmap_w5.png', 'Jadeplant_ssd_dmap_w5.png', 'Motorcycle_ssd_dmap_w5.png', 'Piano_ssd_dmap_w5.png',
            'Pipes_ssd_dmap_w5.png', 'Playroom_ssd_dmap_w5.png', 'Playtable_ssd_dmap_w5.png', 'Recycle_ssd_dmap_w5.png',
            'Shelves_ssd_dmap_w5.png', 'Teddy_ssd_dmap_w5.png', 'Vintage_ssd_dmap_w5.png']

res_w19 = ['Adirondack_ssd_dmap_w19.png', 'Jadeplant_ssd_dmap_w19.png', 'Motorcycle_ssd_dmap_w19.png', 'Piano_ssd_dmap_w19.png',
            'Pipes_ssd_dmap_w19.png', 'Playroom_ssd_dmap_w19.png', 'Playtable_ssd_dmap_w19.png', 'Recycle_ssd_dmap_w19.png',
            'Shelves_ssd_dmap_w19.png', 'Teddy_ssd_dmap_w19.png', 'Vintage_ssd_dmap_w19.png']

res_V0 = ['Adirondack_ae_dmap_V0.png', 'Jadeplant_ae_dmap_V0.png', 'Motorcycle_ae_dmap_V0.png', 'Piano_ae_dmap_V0.png',
            'Pipes_ae_dmap_V0.png', 'Playroom_ae_dmap_V0.png', 'Playtable_ae_dmap_V0.png', 'Recycle_ae_dmap_V0.png',
            'Shelves_ae_dmap_V0.png', 'Teddy_ae_dmap_V0.png', 'Vintage_ae_dmap_V0.png']

res_V1 = ['Adirondack_ae_dmap_V1.png', 'Jadeplant_ae_dmap_V1.png', 'Motorcycle_ae_dmap_V1.png', 'Piano_ae_dmap_V1.png',
            'Pipes_ae_dmap_V1.png', 'Playroom_ae_dmap_V1.png', 'Playtable_ae_dmap_V1.png', 'Recycle_ae_dmap_V1.png',
            'Shelves_ae_dmap_V1.png', 'Teddy_ae_dmap_V1.png', 'Vintage_ae_dmap_V1.png']

res_V2 = ['Adirondack_ae_dmap_V2.png', 'Jadeplant_ae_dmap_V2.png', 'Motorcycle_ae_dmap_V2.png', 'Piano_ae_dmap_V2.png',
            'Pipes_ae_dmap_V2.png', 'Playroom_ae_dmap_V2.png', 'Playtable_ae_dmap_V2.png', 'Recycle_ae_dmap_V2.png',
            'Shelves_ae_dmap_V2.png', 'Teddy_ae_dmap_V2.png', 'Vintage_ae_dmap_V2.png']


print("%-30s%10s%10s%10s%10s%10s" % ('Scene', 'w5', 'w19', 'V0', 'V1', 'V2'))
print('-'*80)

for gt, w5, w19, V0, V1, V2 in zip(gt_imgs, res_w5, res_w19, res_V0, res_V1, res_V2):

    img_gt = cv2.imread(os.path.join(GT_DIR, gt), cv2.IMREAD_GRAYSCALE)
    
    scores = []
    for res in (w5, w19, V0, V1, V2):
        img_res = cv2.imread(os.path.join(RES_DIR, res), cv2.IMREAD_GRAYSCALE)
        scores.append( ssim(img_gt, img_res, multichannel=False) )

    print("%-30s%10.3f%10.3f%10.3f%10.3f%10.3f" % (gt[:-6], scores[0], scores[1], scores[2], scores[3], scores[4]))