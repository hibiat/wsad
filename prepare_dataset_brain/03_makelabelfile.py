""" 
Make the following two csv files by referring to stage_2_train.csv and stage1_train/test_cls.csv

(1) Slice (=frame) label 
studyID, png filename (=studyID_slicenum), epidural,intraparenchymal,intraventricular,subarachnoid,subdural,any

(2) Scan (=video) label
studyID, normal(=0) or anomaly(=1), number of slices
"""

import pandas as pd
from tqdm import tqdm

rootdir = '/path_to/rsna-intracranial-hemorrhage-detection/'
stage1_train_ref = rootdir + 'stage1_train_cls.csv'
stage1_test_ref = rootdir + 'stage1_test_cls.csv'
label_ref = rootdir + 'stage_2_train.csv'
slicelabel_train_out = rootdir + 'slicelabel_stage1_train.csv'
slicelabel_test_out = rootdir + 'slicelabel_stage1_test.csv'
scanlabel_train_out = rootdir + 'scanlabel_stage1_train.csv'
scanlabel_test_out = rootdir + 'scanlabel_stage1_test.csv'


df_label = pd.read_csv(label_ref, header=0)


for dataset in ['test', 'train']:
    if dataset == 'train':
        df = pd.read_csv(stage1_train_ref, header=0, index_col=0)
        f_out = slicelabel_train_out
        v_out = scanlabel_train_out
    if dataset =='test':
        df = pd.read_csv(stage1_test_ref, header=0, index_col=0)
        f_out = slicelabel_test_out
        v_out = scanlabel_test_out

    slicelabel = [['StudyID', 'SliceIndex', 'PngFileName', 'PatientID', \
                'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']]
    scanlabel = pd.DataFrame(index=[], columns=['StudyID', 'ScanLabel', 'NumSlices'])
    with tqdm(total=len(df)) as pbar:
        for index, row in tqdm(df.iterrows()):
            pngfilename_now = row['filename'].split('.', 1)[0] #excluding extension
            slice_id = row['slice_id']
            study_id = row['study_instance_uid']
            study_id_tmp = slice_id.rsplit('_', 1)[0]
            assert study_id == study_id_tmp, 'study_id mismatch. study_id:{} study_id from slice_id {}'.format(study_id, study_id_tmp)
            patient_id = row['patient_id']
            slice_num = slice_id.rsplit('_', 1)[1]
            slice_num = slice_num.zfill(3) #zero padding
            pngfilename_new = study_id + '_' + slice_num +'.png'

            #slice label
            idx = df_label.index[df_label['ID']==pngfilename_now+'_epidural'].values[0] #index of epidural
            epidural = df_label.iat[idx, 1]
            intraparenchymal = df_label.iat[idx + 1, 1]
            intraventricular = df_label.iat[idx + 2, 1]
            subarachnoid = df_label.iat[idx + 3, 1]
            subdural = df_label.iat[idx + 4, 1]
            any = df_label.iat[idx + 5, 1]
            slicelabel.append([study_id, slice_num, pngfilename_new, patient_id, \
                            epidural, intraparenchymal, intraventricular, subarachnoid, subdural, any])
            #scan label
            scan = any
            if sum(scanlabel['StudyID'].str.contains(study_id)) == 0:
                record = pd.Series([study_id, scan, int(slice_num) + 1], index=scanlabel.columns)
                scanlabel = scanlabel.append(record, ignore_index=True)
            else:
                idx = scanlabel.index[scanlabel['StudyID']==study_id].values[0]
                slice_num_now = int(scanlabel.iat[idx, 2])
                if int(slice_num) + 1 > slice_num_now: 
                    scanlabel.iat[idx, 2] = int(slice_num) + 1 #update numslices
                if int(scan) == 1:
                    scanlabel.iat[idx, 1] = 1 #update scanlabel

            pbar.update(1)

        slicelabel_df = pd.DataFrame(slicelabel)
        slicelabel_df.to_csv(f_out, header=False, index=False)
        scanlabel.to_csv(v_out, header=True, index=False)



