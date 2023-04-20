""" 
Make the following two csv files 
- Scan (=video) label
 patientID, normal(=0) or anomaly(=covid=1), number of slices
"""

import os
from glob import glob
import csv

rootdir = '/path_to/COVID-CTset/'

for dataset in ['trainval', 'test']:
    outcsv = os.path.join(rootdir, 'scanlabel_'+dataset+'.csv')
    with open(outcsv, 'a') as f:
        writer = csv.writer(f)
        header = ['StudyID', 'ScanLabel', 'NumSlices']
        writer.writerow(header) 
        for cls in ['normal', 'covid']:   
            for dir in os.listdir(os.path.join(rootdir, cls+'_'+dataset)):
                studyid = dir
                scanlabel = 0 if cls=='normal' else 1
                numslices = len(list(glob(os.path.join(os.path.join(rootdir, cls+'_'+dataset), dir, '*.tif'))))
                writer.writerow([studyid, scanlabel, numslices])    
            



