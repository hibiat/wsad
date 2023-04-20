import argparse
import os 
import glob
import numpy as np

parser = argparse.ArgumentParser(description='WSAD')

parser.add_argument('--path_feature_trainval', type=str, help='path to dir containing features for training and validation[.npy]')
parser.add_argument('--scanlabel_trainval', type=str, help='scan-level annotations (and brain area if any) for training and validation') 

parser.add_argument('--weightdir', type=str, help='path to directory containing .ckpt file')
parser.add_argument('--path_feature_test', type=str, help='path to dir containing features for testing [.npy]')
parser.add_argument('--scanlabel_test', type=str, help='scan-level annotations for testing') 
parser.add_argument('--slicelabel_test', type=str, default=None, help='slice-level annotations only for evaluation') 
parser.add_argument('--thr_conf', type=float, default=0.5, help='Threshold of anomaly score (more than this value is treated as anomaly)')

parser.add_argument('--feature_extracted_model', type=str, default='vit_large_patch16_224_in21k', help='type of feature to be extracted ("resnet50", "vit_large_patch16_224_in21k", vit_small_patch16_224_in21k)') 

parser.add_argument('--k', type=int, default=2, help='value of k')
parser.add_argument('--Lambda', type=str, default='64_1', help='lambda (specify with _ like 64_1 for 64)')

parser.add_argument('--lr', type=float, default=0.0001,help='learning rate')
parser.add_argument('--max_epoch', type=int, default=100, help='maximum iteration to train')
parser.add_argument('--model_name', default='model_mean', help='architecture of network (model_single, model_mean, model_sequence')

parser.add_argument('--brainarea_ratio', type=float, default=0.7, help='Threshold of masked brain area compared to the maximum brain area for feeding a model') 
#Slices containing a brain with "max brain area * this value" are fed into a model. In other words, all slices are fed into model when set to zero
parser.add_argument('--valratio', type=float, default=0.1, help='ratio of valudation out of training data')
parser.add_argument('--sample_size',  type=int, default=30, help='number of normal and anomalous samples in one itration. Actual batchsize is doubled this value')
parser.add_argument('--sample_size_adjust',  type=int, default=25, help='Make normal and anomolous samples imbalance. Add this value to anomolous sample size')
parser.add_argument('--max_slicelen', type=int, default=71, help='maximum number of slices in a batch')
parser.add_argument('--snapshot', type=int, default=100, help='interval of model saving')

parser.add_argument('--device', type=int, default=0, help='GPU ID')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model')

parser.add_argument('--saveroot', type=str, help='path to save directory')







