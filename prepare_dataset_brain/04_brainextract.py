# Detect brain and calcurate the brain area (number of pixels) using brain channel image (should be grayscale) and generate
# - masked image 
# - overlapped image
# - csv file containing brain area of each slice

import os 
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm


rootdir = '/path_to/rsna-intracranial-hemorrhage-detection/'
in_brainch_dir = rootdir + 'stage_1_test_png_brainch' #path to brain channel image
in_scanlabel = rootdir + 'scanlabel_stage1_test.csv'
out_dir = rootdir + 'stage_1_test_brainextract'
thr_max = 254 #max threthhold for binarization
thr_min = 1 #min threthhold for binarization
kernel_size = 9 # kernel size for morphology to identify brain blob


def maskbrain(brainimg, thrmax, thrmin, kernelsize):
    binimg = (brainimg > thrmin) * (brainimg < thrmax)
    binimg = np.asarray(binimg*255.0, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelsize, kernelsize))
    morimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, kernel)
    
    label = cv2.connectedComponentsWithStats(morimg) #https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html
    nblob = label[0] - 1
    if nblob > 0:
        labelimg = label[1]
        stats = np.delete(label[2], 0, 0)
        maxarea_index = np.argmax(stats[:,4]) + 1 #choose the largest blob 
        maskimg = np.where(labelimg==maxarea_index, morimg, 0)
        maskimg = cv2.morphologyEx(maskimg, cv2.MORPH_CLOSE, kernel)
        maskimg = cv2.morphologyEx(maskimg, cv2.MORPH_OPEN, kernel)
        maskimg = cv2.morphologyEx(maskimg, cv2.MORPH_CLOSE, kernel)
        maskimg = cv2.morphologyEx(maskimg, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(maskimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #filling polygon
        for i in range(len(contours)):
            maskimg = cv2.fillPoly(maskimg, pts=[contours[i][:,0,:]], color=(255))
        maskimg = cv2.morphologyEx(maskimg, cv2.MORPH_CLOSE, kernel)
        maskimg = cv2.morphologyEx(maskimg, cv2.MORPH_OPEN, kernel)
    else:
        maskimg = morimg
    return maskimg




if __name__ =='__main__':
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df = pd.read_csv(in_scanlabel, header=0, index_col=None)
    out_list = [['StudyID', 'ScanLabel', 'NumSlices']+list(range(60))] #the maximum number of slices is 60 
    with tqdm(total=len(df)) as pbar:
        for index, row in tqdm(df.iterrows()):
            studyid = row['StudyID']
            scanlabel = row['ScanLabel']
            numslices = row['NumSlices']
            brainarea = np.zeros(60, dtype=np.int64)
            if not os.path.exists(os.path.join(out_dir, studyid)):
                os.makedirs(os.path.join(out_dir, studyid))
            for i in range(numslices):
                filename = os.path.join(in_brainch_dir, studyid, studyid + '_' + str(i).zfill(3) + '.png')
                brainimg = cv2.imread(filename, 0) #grayscale
                maskimg = maskbrain(brainimg=brainimg, thrmax=thr_max, thrmin=thr_min, kernelsize=kernel_size)
                savefilename = os.path.join(out_dir, studyid, studyid + '_' + str(i).zfill(3) + '.png')
                cv2.imwrite(savefilename, maskimg)

                brain_p = np.concatenate([brainimg[:,:,np.newaxis], brainimg[:,:,np.newaxis], brainimg[:,:,np.newaxis]], axis=2)
                mask_p = np.concatenate([0*maskimg[:,:,np.newaxis], 0*maskimg[:,:,np.newaxis], maskimg[:,:,np.newaxis]], axis=2)
                mixture_rate = 0.8
                ovlapimg = cv2.addWeighted(brain_p, mixture_rate, mask_p, 1 - mixture_rate, 0)
                savefilename = os.path.join(out_dir, studyid, studyid + '_' + str(i).zfill(3) + '_ovl.png')
                cv2.imwrite(savefilename, ovlapimg)
                brainarea[i] = np.sum(maskimg > 0) #pixel value of 255 in brain image regers to the brain, value of 0 otherwise
            out_list.append([studyid, scanlabel, numslices] + brainarea.tolist())
            pbar.update(1)

    out_df = pd.DataFrame(out_list)
    out_df.to_csv(os.path.join(out_dir, 'brainarea.csv'), header=False, index=False)