# Make mask images based on brain extracted area
# masked images are all 3ch image even if the original image is 1ch
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

rootdir = '/path_to/rsna-intracranial-hemorrhage-detection/'
in_brainextract_dir = rootdir + 'stage_1_test_brainextract' #path to brain extract images
in_masked_dir = rootdir + 'stage_1_test_png_3ch' #images to be masked
out_dir = rootdir + 'stage_1_test_png_3ch_mask'

with tqdm(total=len(os.listdir(in_brainextract_dir))) as pbar:
    for studyid in os.listdir(in_brainextract_dir):
        if os.path.isdir(studyid):
            numslice = int(len(glob.glob(os.path.join(in_brainextract_dir, studyid,'*.png')))/2) #excluding '*_ovl.png '
            if not os.path.exists(os.path.join(out_dir, studyid)):
                os.makedirs(os.path.join(out_dir, studyid))
            for slice in range(numslice):
                filename = os.path.join(in_brainextract_dir, studyid, studyid+'_'+str(slice).zfill(3)+'.png')
                brainextimg = cv2.imread(filename) #read as 3ch image
                #brainextimg = np.concatenate([brainextimg[:,:,np.newaxis], brainextimg[:,:,np.newaxis], brainextimg[:,:,np.newaxis]], axis=2)
                img = cv2.imread(os.path.join(in_masked_dir, studyid, studyid+'_'+str(slice).zfill(3)+'.png')) #read as 3ch image
                maskimg = np.where(brainextimg == 0, 0, img)
                cv2.imwrite(os.path.join(out_dir, studyid, studyid+'_'+str(slice).zfill(3)+'.png'), maskimg)

        pbar.update(1)