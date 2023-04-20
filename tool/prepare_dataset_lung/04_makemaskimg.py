# Just copying existing mask images (body, lung)

from email.mime import base
import os
import shutil
from glob import glob

rootdir = '/path_to/COVID-CTset/'

for data in ['trainval', 'test']:
    indir = rootdir+data+'_extract'
    for patient in os.listdir(indir):
        outdir_nomask = os.path.join(rootdir,data+'_nomask', patient)
        outdir_maskbody = os.path.join(rootdir,data+'_maskbody', patient)
        outdir_masklung = os.path.join(rootdir,data+'_masklung', patient)
        if not os.path.exists(outdir_nomask):
            os.makedirs(outdir_nomask)
        if not os.path.exists(outdir_maskbody):
            os.makedirs(outdir_maskbody)
        if not os.path.exists(outdir_masklung):
            os.makedirs(outdir_masklung)
        
        for file in glob(os.path.join(indir, patient, '*_00original.png')):
            filenamenow = os.path.join(indir, patient, file)
            basenamenew = os.path.basename(file).split('_00original',1)[0]+'.png'
            filenamenew = os.path.join(outdir_nomask, basenamenew)
            shutil.copy(filenamenow, filenamenew)
        for file in glob(os.path.join(indir, patient, '*_03maskbody.png')):
            filenamenow = os.path.join(indir, patient, file)
            basenamenew = os.path.basename(file).split('_03maskbody',1)[0]+'.png'
            filenamenew = os.path.join(outdir_maskbody, basenamenew)
            shutil.copy(filenamenow, filenamenew)
        for file in glob(os.path.join(indir, patient, '*_07masklunghigh.png')):
            filenamenow = os.path.join(indir, patient, file)
            basenamenew = os.path.basename(file).split('_07masklunghigh',1)[0]+'.png'
            filenamenew = os.path.join(outdir_masklung, basenamenew)
            shutil.copy(filenamenow, filenamenew)
    
        
    