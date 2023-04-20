# Detect lung and calcurate the lung area (number of pixels)  and generate
# - masked image 
# - overlapped image
# - csv file containing lung area of each slice

import os 
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import PIL.Image as pil_image
import io


rootdir = '/path_to/COVID-CTset/'
thr_body = 10 #threthhold for binarization for body extraction
kernel_size_body = 15 # kernel size for morphology to identify blob for body extraction
thr_lung = 60 #  for lung extraction
kernel_size_lung = 13



def extractbody(img, thr, kernel_size):
    #binimg = (img > thrmin) * (img < thrmax)
    binimg = img > thr
    binimg = np.asarray(binimg*255.0, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    morimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, kernel)

    label = cv2.connectedComponentsWithStats(morimg) #https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html
    nblob = label[0] - 1
    if nblob > 0:
        labelimg = label[1]
        stats = np.delete(label[2], 0, 0)
        maxarea_index = np.argmax(stats[:,4]) + 1 #choose the largest blob 
        morimg = np.where(labelimg==maxarea_index, morimg, 0)
        morimg = cv2.morphologyEx(morimg, cv2.MORPH_OPEN, kernel)
        morimg = cv2.morphologyEx(morimg, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(morimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #filling polygon
        for i in range(len(contours)):
            morimg = cv2.fillPoly(morimg, pts=[contours[i][:,0,:]], color=(255))
        morimg = cv2.morphologyEx(morimg, cv2.MORPH_CLOSE, kernel)
        morimg = cv2.morphologyEx(morimg, cv2.MORPH_OPEN, kernel)
    maskimg = np.where(morimg > 0, img, 0)
    return morimg, maskimg

def extractlung(img, bodymaskimg, thr, kernel_size):
    binimg = (img < thr) * (bodymaskimg > 0)
    binimg = np.asarray(binimg*255.0, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    morimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, kernel)
    morimg = cv2.morphologyEx(morimg, cv2.MORPH_CLOSE, kernel)
    # morimg = cv2.morphologyEx(morimg, cv2.MORPH_CLOSE, kernel)
    # morimg = cv2.morphologyEx(morimg, cv2.MORPH_CLOSE, kernel)
    # morimg = cv2.morphologyEx(morimg, cv2.MORPH_CLOSE, kernel)
    
    label = cv2.connectedComponentsWithStats(morimg) #https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html
    nblob = label[0] - 1
    if nblob > 0:
        labelimg = label[1]
        stats = np.delete(label[2], 0, 0)
        maxarea_index = np.argmax(stats[:,4]) + 1 #choose the largest blob 
        if nblob == 1:
            morimg = np.where(labelimg==maxarea_index, morimg, 0)
        else:
            max_area_index2 = np.argsort(-stats[:,4])[1] + 1 #choose the 2nd largest blob 
            # if there is no ovarlap in y axis between bbox of the 2nd largest blob and the largest blob, it is neglected
            y1 = stats[maxarea_index-1,1]
            h1 = stats[maxarea_index-1,3]
            y2 = stats[max_area_index2-1,1]
            h2 = stats[max_area_index2-1,3]
            if len(set(range(y1, y1+h1+1)) & set(range(y2, y2+h2+1)))>0:
                morimg = np.where((labelimg==maxarea_index) | (labelimg==max_area_index2), morimg, 0)
            else:
                morimg = np.where(labelimg==maxarea_index, morimg, 0) # No overlap
                           
        contours, _ = cv2.findContours(morimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #filling polygon
        for i in range(len(contours)):
            morimg = cv2.fillPoly(morimg, pts=[contours[i][:,0,:]], color=(255))
            
        # morimg = cv2.dilate(morimg, kernel, iterations=1)
        # morimg = cv2.erode(morimg, kernel, iterations=1)
        morimg = cv2.morphologyEx(morimg, cv2.MORPH_OPEN, kernel)
        morimg = cv2.morphologyEx(morimg, cv2.MORPH_CLOSE, kernel)
        morimg = cv2.morphologyEx(morimg, cv2.MORPH_OPEN, kernel)
        morimg = cv2.morphologyEx(morimg, cv2.MORPH_CLOSE, kernel)
        # morimg = cv2.morphologyEx(morimg, cv2.MORPH_CLOSE, kernel)
        # morimg = cv2.morphologyEx(morimg, cv2.MORPH_CLOSE, kernel)

    maskimg = np.where(morimg > 0, img, 0)
    return morimg, maskimg

def tif8bit(tifimg):
    with open(tifimg, 'rb') as f:
        tif = pil_image.open(io.BytesIO(f.read()))
        array=np.array(tif)
        max_val=np.amax(array)
        if max_val>0:
            normalized=np.array(255.0 * array/max_val, dtype=np.uint8)
        else:
            normalized = np.zeros(array.shape, dtype=np.uint8)
    return normalized

def tif8bitmask(tifimg, maskarea): #extract mask area from 16bit tiff and normalize it
    with open(tifimg, 'rb') as f:
        tif = pil_image.open(io.BytesIO(f.read()))
        array=np.array(tif)
        array_h = np.where(maskarea>0, array, 0.0)
        max_val=np.amax(array_h)
        if max_val>0:
            normalized=np.array(255.0 * array_h/max_val, dtype=np.uint8)
        else:
            normalized = np.zeros(array.shape, dtype=np.uint8)
    return normalized

if __name__ =='__main__':
    for data in ['trainval', 'test']:
        in_img_dir = rootdir + data #path to CT image
        in_scanlabel = rootdir + 'scanlabel_'+data+'.csv'
        out_dir = rootdir + data+'_extract'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        df = pd.read_csv(in_scanlabel, header=0, index_col=None)
        out_list = [['StudyID', 'ScanLabel', 'NumSlices']+list(range(71))] #the maximum number of slices is 71 
        with tqdm(total=len(df)) as pbar:
            for index, row in tqdm(df.iterrows()):
                studyid = row['StudyID']
                scanlabel = row['ScanLabel']
                numslices = row['NumSlices']
                lungarea = np.zeros(71, dtype=np.int64)
                if not os.path.exists(os.path.join(out_dir, studyid)):
                    os.makedirs(os.path.join(out_dir, studyid))
                for i in range(numslices):
                    filename = os.path.join(in_img_dir, studyid, studyid + '_' + str(i).zfill(3) + '.tif')
                    img = tif8bit(filename) #grayscale

                    savefilename = os.path.join(out_dir, studyid, studyid + '_' + str(i).zfill(3) + '_00original.png')
                    cv2.imwrite(savefilename, img)

                    morimg, maskimg = extractbody(img=img, thr=thr_body, kernel_size=kernel_size_body)
                    savefilename = os.path.join(out_dir, studyid, studyid + '_' + str(i).zfill(3) + '_03maskbody.png')
                    cv2.imwrite(savefilename, maskimg)

                    savefilename = os.path.join(out_dir, studyid, studyid + '_' + str(i).zfill(3) + '_01binbody.png')
                    cv2.imwrite(savefilename, morimg)

                    brain_p = np.concatenate([img[:,:,np.newaxis], img[:,:,np.newaxis], img[:,:,np.newaxis]], axis=2)
                    mor_p = np.concatenate([0*morimg[:,:,np.newaxis], 0*morimg[:,:,np.newaxis], morimg[:,:,np.newaxis]], axis=2)
                    mixture_rate = 0.7
                    ovlapimg = cv2.addWeighted(brain_p, mixture_rate, mor_p, 1 - mixture_rate, 0)
                    savefilename = os.path.join(out_dir, studyid, studyid + '_' + str(i).zfill(3) + '_02ovlbody.png')
                    cv2.imwrite(savefilename, ovlapimg)

                    morimg_lung, maskimg_lung = extractlung(img=maskimg, bodymaskimg=morimg, thr=thr_lung, kernel_size=kernel_size_lung)

                    savefilename = os.path.join(out_dir, studyid, studyid + '_' + str(i).zfill(3) + '_06masklung.png')
                    cv2.imwrite(savefilename, maskimg_lung)

                    maskimg_lung_hcontrast = tif8bitmask(filename, morimg_lung)
                    savefilename = os.path.join(out_dir, studyid, studyid + '_' + str(i).zfill(3) + '_07masklunghigh.png')
                    cv2.imwrite(savefilename, maskimg_lung_hcontrast)


                    savefilename = os.path.join(out_dir, studyid, studyid + '_' + str(i).zfill(3) + '_04binlung.png')
                    cv2.imwrite(savefilename, morimg_lung)

                    brain_p = np.concatenate([maskimg[:,:,np.newaxis], maskimg[:,:,np.newaxis], maskimg[:,:,np.newaxis]], axis=2)
                    mor_p = np.concatenate([0*morimg_lung[:,:,np.newaxis], 0*morimg_lung[:,:,np.newaxis], morimg_lung[:,:,np.newaxis]], axis=2)
                    mixture_rate = 0.7
                    ovlapimg = cv2.addWeighted(brain_p, mixture_rate, mor_p, 1 - mixture_rate, 0)
                    savefilename = os.path.join(out_dir, studyid, studyid + '_' + str(i).zfill(3) + '_05ovllung.png')
                    cv2.imwrite(savefilename, ovlapimg)
                
                    lungarea[i] = np.sum(morimg_lung > 0) 
                out_list.append([studyid, scanlabel, numslices] + lungarea.tolist())
                pbar.update(1)

        out_df = pd.DataFrame(out_list)
        out_df.to_csv(os.path.join(out_dir, 'lungarea'+data+'.csv'), header=False, index=False)