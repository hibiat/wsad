import os
import glob
import numpy as np


def checkfeaturesize(args):
    featurefile_trainval = glob.glob(os.path.join(args.path_feature_trainval, args.feature_extracted_model, '*.npy'))[0]
    feature_trainval = np.load(featurefile_trainval)
    featurefile_test = glob.glob(os.path.join(args.path_feature_test, args.feature_extracted_model, '*.npy'))[0]
    feature_test = np.load(featurefile_test)
    assert feature_trainval.shape[1]==feature_test.shape[1], f'dimension of feature for train is{feature_trainval.shape[1]}, but for test is{feature_test.shape[1]}'
    feature_size = feature_trainval.shape[1]
    print('{} dimension features are extracted from {}'.format(feature_size, args.feature_extracted_model))
    return feature_size


def read_dataset(dataset):
    feature = dataset['feature']
    scanlabel = dataset['scanlabel']
    slicelabel = dataset['slicelabel']
    studyid = dataset['studyid']
    numslices = dataset['numslices']
    slice_start = dataset['slice_start']
    slice_end = dataset['slice_end']
    return feature, scanlabel, slicelabel, studyid, numslices, slice_start, slice_end