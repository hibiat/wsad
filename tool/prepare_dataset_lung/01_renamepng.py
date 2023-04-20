"""
Create directory of patient ID and save tiff files in them 
in accordance with brain CT images
"""
import os
from glob import glob
import shutil
import math

rootdir = '/path_to/COVID-CTset/'
indir_covid = rootdir + '_original_covid'
indir_normal = rootdir + '_original_normal'
outdir_covid = rootdir + 'covid_all'
outdir_normal = rootdir + 'normal_all'

for dataset in ['covid', 'normal']:
    if dataset == 'covid':
        indir = indir_covid
        outdir= outdir_covid
        print('Covid dataset')
    if dataset == 'normal':
        indir = indir_normal
        outdir= outdir_normal
        print('Normal dataset')
    
    for patient in os.listdir(indir):
        assert len(list(glob(os.path.join(indir, patient, 'SR_2', '*.tif'))))>0, f'No tiff file found in {patient}'
        for count, file in enumerate(sorted(glob(os.path.join(indir, patient, 'SR_2', '*.tif')))):
            patientid = 'patient' + patient.split('patient',1)[1].zfill(3) #'patient1' => 'patient001'
            sliceid = str(count).zfill(3) # 0 start, IM00001 => 001 
            filename_now = os.path.join(indir, patient,'SR_2',file)
            filename_new = os.path.join(outdir, patientid, patientid+'_'+ sliceid + '.tif')
            if not os.path.exists(os.path.join(outdir, patientid)):
                os.makedirs(os.path.join(outdir, patientid))
            shutil.copy(filename_now, filename_new)

