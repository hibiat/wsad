from operator import index
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd

from utils import read_dataset


def savegraph(element_logits, slicelabel, studyid, original_numslices, slice_start, slice_end, save_path):
    x_all = np.arange(original_numslices)
    y_all = np.zeros(int(original_numslices))
    y_all[slice_start:slice_end + 1] = element_logits
    fig = plt.figure()
    for i in range(len(x_all)):
        plt.plot(x_all[i:i+2], y_all[i:i+2], linewidth=2, color='b' if x_all[i]>=slice_start-1 and x_all[i]<=slice_end else 'b')
    
    if int(torch.sum(slicelabel==-1))!=slicelabel.size()[1]:
        slicelabel =  np.array(slicelabel).reshape(slicelabel.size()[1])
        plt.fill_between(x_all, 0, 1, step='mid', facecolor="blue", alpha=0.15)
        plt.fill_between(x_all, 0, 1, where=slicelabel > 0, step='mid', facecolor="red", alpha=0.3)
    plt.xticks(np.arange(0, original_numslices, step=5))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xlabel('CT Slices')
    plt.ylabel('Anomaly scores')
    plt.grid(True, linestyle='-.')
    plt.ylim(-0.05, 1.05)
    plt.savefig(os.path.join(save_path, str(studyid)+'.png'))
    plt.close()



def test(test_normal_loader, test_anomaly_loader, model, device, save_path, args):
    model.eval()
    if not os.path.exists(os.path.join(save_path, 'result')):
        os.makedirs(os.path.join(save_path, 'result'))
    #normal samples
    with torch.no_grad():
        for data in tqdm(test_normal_loader, desc='test_normal', total=len(test_normal_loader)):
            features, scanlabels, slicelabel , studyid , original_numslices, slice_start, slice_end = read_dataset(data)
            scanlabels = scanlabels[:,None].float().to(device)
            numslices = slice_end - slice_start + 1
            maxnumslice = max(numslices)
            features = features[:, :maxnumslice, :].float().to(device)
            if args.model_name == 'model_lstm':
                _, element_logits = model(features, maxnumslice, is_training=False)
            else:
                _, element_logits = model(features, is_training=False)
            element_logits = element_logits.cpu().data.numpy().reshape(-1)[:numslices]
            slicelabel = slicelabel.cpu().data.numpy().reshape(-1)[:original_numslices]
            slicelabel = np.squeeze(slicelabel)
            savegraph(element_logits, torch.tensor(slicelabel).unsqueeze(0), str(studyid[0]), original_numslices, slice_start, slice_end, os.path.join(save_path, 'result'))


    #anomaly samples
    with torch.no_grad():
        for data in tqdm(test_anomaly_loader, desc='test_anomaly', total=len(test_anomaly_loader)):
            features, scanlabels, slicelabel , studyid , original_numslices, slice_start, slice_end = read_dataset(data)
            scanlabels = scanlabels[:,None].float().to(device)
            numslices = slice_end - slice_start + 1
            maxnumslice = max(numslices)
            features = features[:, :maxnumslice, :].float().to(device)
            if args.model_name == 'model_lstm':
                _, element_logits = model(features, maxnumslice, is_training=False)
            else:
                _, element_logits = model(features, is_training=False)
            element_logits = element_logits.cpu().data.numpy().reshape(-1)[:numslices]
            slicelabel = slicelabel.cpu().data.numpy().reshape(-1)[:original_numslices]
            slicelabel = np.squeeze(slicelabel)
            savegraph(element_logits, torch.tensor(slicelabel).unsqueeze(0), str(studyid[0]), original_numslices, slice_start, slice_end, os.path.join(save_path, 'result'))



def test_no_slicelabel(test_normal_loader, test_anomaly_loader, model, device, save_path, args):
    model.eval()
    if not os.path.exists(os.path.join(save_path, 'result')): 
        os.makedirs(os.path.join(save_path, 'result')) 
    #normal samples
    with torch.no_grad():
        for data in tqdm(test_normal_loader, desc='test_normal', total=len(test_normal_loader)):
            features, scanlabels, slicelabel , studyid , original_numslices, slice_start, slice_end = read_dataset(data)
            scanlabels = scanlabels[:,None].float().to(device)
            numslices = slice_end - slice_start + 1 
            maxnumslice = max(numslices)
            features = features[:, :maxnumslice, :].float().to(device)
            if args.model_name == 'model_lstm':
                _, element_logits = model(features, maxnumslice, is_training=False)
            else:
                _, element_logits = model(features, is_training=False)
            element_logits = element_logits.cpu().data.numpy().reshape(-1)[:numslices]
            savegraph(element_logits, slicelabel, str(studyid[0]), original_numslices, slice_start, slice_end, os.path.join(save_path, 'result')) 

    #anomaly samples
    with torch.no_grad():
        for data in tqdm(test_anomaly_loader, desc='test_anomaly', total=len(test_anomaly_loader)):
            features, scanlabels, slicelabel , studyid , original_numslices, slice_start, slice_end = read_dataset(data)
            scanlabels = scanlabels[:,None].float().to(device)
            numslices = slice_end - slice_start + 1
            maxnumslice = max(numslices)
            features = features[:, :maxnumslice, :].float().to(device)
            if args.model_name == 'model_lstm':
                _, element_logits = model(features, maxnumslice, is_training=False)
            else:
                _, element_logits = model(features, is_training=False)
            element_logits = element_logits.cpu().data.numpy().reshape(-1)[:numslices]
            savegraph(element_logits, slicelabel, str(studyid[0]), original_numslices, slice_start, slice_end, os.path.join(save_path, 'result')) 

 
