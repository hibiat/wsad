from hashlib import new
import joblib
import PIL
from glob import glob
import pydicom
import numpy as np
import pandas as pd
import os
import cv2
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image
import math
import seaborn as sns
from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm
import re
import logging as l
from glob import glob
import argparse

np.seterr(divide='ignore', invalid='ignore')

def get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    return int(x)

def get_id(img_dicom):
    return str(img_dicom.SOPInstanceUID)

def get_metadata_from_dicom(img_dicom):
    metadata = {
        "window_center": img_dicom.WindowCenter,
        "window_width": img_dicom.WindowWidth,
        "intercept": img_dicom.RescaleIntercept,
        "slope": img_dicom.RescaleSlope,
    }
    return {k: get_first_of_dicom_field_as_int(v) for k, v in metadata.items()}


def window_image(img, window_center, window_width, intercept, slope):
    img = img * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img 


def three_window_image(img, window_center, window_width, intercept, slope):
    brain_window = window_image(img, 40, 80, intercept, slope)
    brain_window = normalize_minmax(brain_window)
    subdural_window = window_image(img, 80, 200, intercept, slope)
    subdural_window = normalize_minmax(subdural_window)
    bone_window= window_image(img, 600, 2800, intercept, slope)
    bone_window = normalize_minmax(bone_window)
    image_cat = np.concatenate([brain_window[:,:,np.newaxis], subdural_window[:,:,np.newaxis], bone_window[:,:,np.newaxis]],2)
    return image_cat

def brainwindow_image(img, window_center, window_width, intercept, slope):
    brain_window = window_image(img, 40, 80, intercept, slope)
    brain_window = normalize_minmax(brain_window)
    return brain_window 

def bonewindow_image(img, window_center, window_width, intercept, slope):
    bone_window = window_image(img, 600, 2800, intercept, slope)
    bone_window = normalize_minmax(bone_window)
    return bone_window 

def resize(img, new_w, new_h):
    img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    return img.resize((new_w, new_h), resample=PIL.Image.BICUBIC)

def save_img(img_pil, subfolder, name):
    img_pil.save(subfolder+name+'.png')

def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    return (img - mi) / (ma - mi)

def prepare_image(img_path, numwindow):
    img_dicom = pydicom.read_file(img_path)
    img_id = get_id(img_dicom)
    metadata = get_metadata_from_dicom(img_dicom)
    if numwindow == 1:
        img = window_image(img_dicom.pixel_array, **metadata)
        img = normalize_minmax(img) * 255
        img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    elif numwindow == 3:
        img = three_window_image(img_dicom.pixel_array, **metadata)
        img = img * 255
        img = PIL.Image.fromarray(img.astype(np.int8), mode="RGB")
    elif numwindow == 4:
        img = brainwindow_image(img_dicom.pixel_array, **metadata)
        img = normalize_minmax(img) * 255
        img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    elif numwindow == 5:
        img = bonewindow_image(img_dicom.pixel_array, **metadata)
        img = normalize_minmax(img) * 255
        img = PIL.Image.fromarray(img.astype(np.int8), mode="L")
    else:
        raise NotImplementedError()
    return img_id, img

def prepare_and_save(img_path, subfolder, numwindow):
    try:
        img_id, img_pil = prepare_image(img_path, numwindow)
        save_img(img_pil, subfolder, img_id)
    except KeyboardInterrupt:
        # Rais interrupt exception so we can stop the cell execution
        # without shutting down the kernel.
        raise
    except:
        l.error('Error processing the image: {'+img_path+'}')

def prepare_images(imgs_path, subfolder):
    for i in tqdm.tqdm(imgs_path):
        prepare_and_save(i, subfolder)

def prepare_images_njobs(img_paths, subfolder, numwindow=1, n_jobs=-1):
    joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(prepare_and_save)(i, subfolder, numwindow) for i in tqdm(img_paths))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-dcm_path", "--dcm_path", type=str)
    parser.add_argument("-png_path", "--png_path", type=str)
    parser.add_argument("-numwindow", "--numwindow", type=int, default=1, help='(1)all-into-single window or (3)3-window into RGB(brain, subdural, bone) images ,(4)brain window (5) bone window')
    
    args = parser.parse_args()
    dcm_path = args.dcm_path
    png_path = args.png_path
    numwindow = args.numwindow

    if not os.path.exists(png_path):
        os.makedirs(png_path)

    prepare_images_njobs(glob(dcm_path+'/*'), png_path+'/', numwindow)
