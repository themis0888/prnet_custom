"""
CUDA_VISIBLE_DEVICES=1 python demo.py \
-i /shared/data/celeb_cartoon/sample/ \
--isDlib True \
--isKpt True --isShow True --isImage True 

CUDA_VISIBLE_DEVICES=0 python demo.py -i /shared/data/sample/ \
-o /shared/data/sample/prnet_out/ --isDlib True \
--isKpt True --isPose True --isShow True --isImage True 
"""
import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast

from api import PRN

from utils.estimate_pose import estimate_pose
from utils.rotate_vertices import frontalize
from utils.render_app import get_visibility, get_uv_mask, get_depth_image
from utils.write import write_obj_with_colors, write_obj_with_texture


parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')

parser.add_argument('-i', '--inputDir', default='TestImages/', type=str,
                    help='path to the input directory, where input images are stored.')
parser.add_argument('-o', '--outputDir', default='TestImages/results', type=str,
                    help='path to the output directory, where results(obj,txt files) will be stored.')
parser.add_argument('--gpu', default='0', type=str,
                    help='set gpu id, -1 for CPU')
args = parser.parse_args()


def plot_kpt(image, kpt):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)
    for i in range(kpt.shape[0]):
        st = kpt[i, :2]
        image = cv2.circle(image,(st[0], st[1]), 1, (120,120,220), 2)  
        if i in end_list:
            continue
        ed = kpt[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)
    return image



cv2.imwrite('sample_go.jpg', plot_kpt(imm, a))