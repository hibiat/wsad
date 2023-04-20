import numpy as np
from torch.utils.data import Dataset, DataLoader
import parameters
import os
import pandas as pd
import torch

def getslicerange(brainarea, brainarea_ratio):
    brain_valid = np.where(brainarea>np.max(brainarea) * brainarea_ratio)
    slice_start = np.min(brain_valid)
    slice_end = np.max(brain_valid)
    return slice_start, slice_end

class dataset(Dataset):
    def __init__(self, path_feature, scanlabel, slicelabel, classname, feature_extracted_model, max_slicelen, brainarea_ratio):
        self.path_feature = path_feature
        self.scanlabel = scanlabel
        self.slicelabel = slicelabel
        self.classname = classname
        self.feature_extracted_model = feature_extracted_model
        self.max_slicelen = max_slicelen
        self.brainarea_ratio = brainarea_ratio

        if self.slicelabel is None:           
            self.df_slice = None             
        else:
            self.df_slice = pd.read_csv(self.slicelabel)    
        self.df = pd.read_csv(self.scanlabel)
        if self.classname == 'normal':
            self.df = self.df[self.df['ScanLabel']==0]
        elif self.classname =='anomaly':
            self.df = self.df[self.df['ScanLabel']==1]
        self.df.reset_index(drop=True, inplace=True)

    def __getitem__(self, index):
        studyid = str(self.df['StudyID'][index])
        scanlabel = int(self.df['ScanLabel'][index])
        numslices = int(self.df['NumSlices'][index])
        featurefile = os.path.join(self.path_feature, self.feature_extracted_model, studyid+'.npy')
        ext_feature = np.load(featurefile)
        feature_dim = ext_feature.shape[1]
        assert ext_feature.shape[0] == numslices, print('extracted feature has {} slices, but should have {} slices'.format(ext_feature.shape[0], numslices))
        feature= np.zeros((self.max_slicelen, feature_dim), dtype=np.float32)
        brainarea = np.array(self.df.loc[index, '0' : str(self.max_slicelen - 1)])
        slice_start, slice_end = getslicerange(brainarea, self.brainarea_ratio)
        feature[ : slice_end - slice_start + 1, :] = ext_feature[slice_start : slice_end + 1, :]

        slicelabel = -1 * torch.ones(self.max_slicelen, dtype=torch.int64) # initialize by -1. If w_slicelabel==False, initialized slicelabel is returned
        if self.slicelabel is not None:
            if self.classname == 'normal':
                slicelabel[0 : numslices] = 0
            elif self.classname == 'anomaly':
                for numslice in range(numslices):
                    pngfilename = studyid+'_'+str(numslice).zfill(3)+'.png'
                    slicelabel[numslice] = int(self.df_slice[self.df_slice['PngFileName']==pngfilename]['any']) # 0 for normal, 1 for anomaly

        return {'feature':feature, 'scanlabel':scanlabel, 'slicelabel':slicelabel, 'studyid':studyid, 'numslices':numslices, 'slice_start':slice_start, 'slice_end':slice_end} 
        #shape: feature: [batch,60slices,2048etc], slicelabel:[batch,60slices], scanlabel,scanlabel,numslices,slice_start: [batch]

    def __len__(self):
        return len(self.df)



if __name__ == "__main__":
    args = parameters.parser.parse_args()
    customdataset = dataset(path_feature=args.path_feature_test,  \
                            scanlabel=args.scanlabel_test, \
                            slicelabel=args.slicelabel_test, \
                            classname='normal', \
                            feature_extracted_model=args.feature_extracted_model, max_slicelen=args.max_slicelen,\
                            brainarea_ratio=args.brainarea_ratio)
    customdataloader = DataLoader(dataset=customdataset, batch_size=3, pin_memory=True,
                              num_workers=5, shuffle=False)
    for batch in customdataloader:
        feature = batch['feature']
        scanlabel = batch['scanlabel']
        slicelabel = batch['slicelabel']
        studyid = batch['studyid']
        numslices = batch['numslices']
        print('Study ID:{}'.format(studyid))
        print('Num Slice:{}'.format(numslices))
        print('Scanlabel: {}'.format(scanlabel))
        print('Slicelabel: {}'.format(slicelabel))
        print('Shape of feature ({})'.format(feature.shape)) #[batch_size, args.max_slicelen, feature dimenstion]
        print(f'slice_start: {batch["slice_start"]}, slice_end: {batch["slice_end"]}')
        print('------')
        