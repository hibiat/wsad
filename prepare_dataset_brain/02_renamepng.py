""" 
Make the following structure by referring to stage1_train/test_cls.csv
Under the assumption that stage_2_train (=stage_1_train + stage_1_test) are converted into png

train
    |
    studyID1 -----studyID1_001.png
    |           |-studyID1_002.png
    |           |-   ...
    |
    studyID2 -----studyID2_001.png
                |-studyID2_002.png
                |-   ...
     ...
test
    |
    (same as above)
"""
import os
import pandas as pd
import shutil

rootdir = '/path_to/rsna-intracranial-hemorrhage-detection/'
stage1_train_ref = rootdir + 'stage1_train_cls.csv'
stage1_test_ref = rootdir + 'stage1_test_cls.csv'
stage_2_train_png = rootdir + 'stage_2_train_png_bonech'
stage1_train_outdir = rootdir + 'stage_1_train_png_bonech'
stage1_test_outdir =  rootdir + 'stage_1_test_png_bonech'

for dataset in ['train', 'test']:
    if dataset == 'train':
        newdir = stage1_train_outdir
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        df = pd.read_csv(stage1_train_ref, header=0, index_col=0)

    if dataset =='test':
        newdir = stage1_test_outdir
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        df = pd.read_csv(stage1_test_ref, header=0, index_col=0)

    for index, row in df.iterrows():
        pngfilename_now = row['filename']
        slice_id = row['slice_id']
        study_id = row['study_instance_uid']
        study_id_tmp = slice_id.rsplit('_', 1)[0]
        assert study_id == study_id_tmp, 'study_id mismatch. study_id:{} study_id from slice_id {}'.format(study_id, study_id_tmp)
        slice_num = slice_id.rsplit('_', 1)[1]
        slice_num = slice_num.zfill(3) #zero padding
        pngfilename_new = study_id + '_' + slice_num +'.png'
        movedir = os.path.join(newdir, study_id)
        if not os.path.exists(movedir):
            os.makedirs(movedir)
        shutil.copy(os.path.join(stage_2_train_png, pngfilename_now), os.path.join(movedir, pngfilename_new))


