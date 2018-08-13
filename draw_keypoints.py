"""
python draw_keypoints.py 
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
import cv2

from api import PRN
import PIL.Image as im


parser = argparse.ArgumentParser(description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')
parser.add_argument('--data_path', type=str, dest='data_path', default='/shared/data/meta/anime/')
parser.add_argument('--data_name', type=str, dest='data_name', default='danbooru')
parser.add_argument('--save_path', type=str, dest='save_path', default='/shared/data/meta/anime/pix2pix/')
parser.add_argument('--meta_path', type=str, dest='meta_path', default='/shared/data/meta/anime/meta/')
config, unparsed = parser.parse_known_args() 

if not os.path.exists(config.save_path):
    os.mkdir(config.save_path)

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
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
        image = cv2.circle(image,(st[0], st[1]), 1, (120,220,80), 2)  
        if i in end_list:
            continue
        ed = kpt[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)
    return image


def file_list(path, extensions, sort=True, path_label = False):
    if path_label == True:
        result = [(os.path.join(dp, f) + ' ' + os.path.join(dp, f).split('/')[-2])
        for dp, dn, filenames in os.walk(path) 
        for f in filenames if os.path.splitext(f)[1] in extensions]
    else:
        result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) 
        for f in filenames if os.path.splitext(f)[1] in extensions]
    if sort:
        result.sort()

    return result

# /shared/data/sample/prnet_out/meta/597447_7_cut_018_skpt_kpt.npy
# /shared/data/sample/597447_7_cut_018.jpg
# /shared/data/meta/anime/03851_004.jpg

# image_path_list = file_list(config.data_path, ('.jpg', '.png'))
image_path_list = ['/shared/data/meta/anime/03851_004.jpg']

for i, image_path in enumerate(image_path_list):

    name = image_path.strip().split('/')[-1][:-4]
    print(image_path)
    # read image
    image = imread(image_path)
    [h, w, c] = image.shape
    b, g, r = cv2.split(image)
    rgb_im = cv2.merge([r,g,b])

    filename = os.path.basename(image_path)
    if not os.path.exists('{}_{}.jpg'.format(h,w)):
        template = im.new('RGB', (2*w,h), color = 'white')
        template.save('{}_{}.jpg'.format(h,w), 'JPEG')

    template = cv2.imread('{}_{}.jpg'.format(h,w))

    for c in range(3):
        template[0:h,0:w,c] = rgb_im[0:h,0:w,c]

    kpt = np.load(os.path.join(config.meta_path,filename[:-4]+'_kpt.npy'))
    kpt[:, 0] += w
    cv2.imwrite(os.path.join(config.save_path, filename), plot_kpt(template, kpt))


