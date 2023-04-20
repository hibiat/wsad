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

def calcmetrics(y_true, y_pred, thr):
    if y_true.count(1) == len(y_true) or y_true.count(0) == len(y_true): # auc cannot be computed in a single class 
        auc = None
    else:
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
    
    bin_pred = [1 if x > thr else 0 for x in y_pred] #1=anomaly=positive, 0=normal=negative
    confm = confusion_matrix(y_true=y_true, y_pred=bin_pred, labels=[0,1])
    tn, fp, fn, tp = confm.flatten()
    if tp + fn > 0:
        sensitivity_recall = tp / (tp + fn)
    else:
        sensitivity_recall = None

    if tn + fp > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = None
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = None
    
    if tp + fp + fn > 0:
        f1 = 2 * tp / (2 * tp + fp + fn)
    else:
        f1 = None
    
    accuracy = (tp + tn)/ (tn + fp + fn + tp)

    return [confm, auc, sensitivity_recall, specificity, precision, f1, accuracy]


def test(test_normal_loader, test_anomaly_loader, model, device, save_path, args):
    model.eval()
    if not os.path.exists(os.path.join(save_path, 'result')):
        os.makedirs(os.path.join(save_path, 'result'))
    #normal samples
    c=0
    pred_normal = [] #slice-level prediction excluding <slice_start and >slice_end
    gt_normal = [] #slice-level groundtruth excluding <slice_start and >slice_end
    pred_all_normal = [] #including <slice_start and >slice_end
    gt_all_normal = [] 

    pred_scan_normal = [] # scan-level prediction = maximum of anomaly scores throughout all slices excluding <slice_start and >slice_end
    gt_scan_normal = [] # scan-level groundtruth  excluding <slice_start and >slice_end
    # If one or more than one slice has an anomaly socre more than <thr>, then the scan-level groundtruth  is anomolous. 
    # If all slices have anomaly score less than <thr>, then the scan-level groundtruth is normal.
    pred_all_scan_normal = [] # including <slice_start and >slice_end
    gt_all_scan_normal = []

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
            pred_normal.extend(element_logits.tolist())
            gt_normal.extend(slicelabel[slice_start:slice_end + 1].tolist())
            numslices.data.cpu().detach().numpy()

            pred_all_normal.extend([0]*slice_start + element_logits.tolist() + [0]*(original_numslices - slice_end - 1)) 
            gt_all_normal.extend(slicelabel[:original_numslices].tolist())
            savegraph(element_logits, torch.tensor(slicelabel).unsqueeze(0), str(studyid[0]), original_numslices, slice_start, slice_end, os.path.join(save_path, 'result'))

            pred_scan_normal.extend([np.max(element_logits)])
            gt_scan_normal.extend([np.max(slicelabel[slice_start:slice_end + 1])])
            pred_all_scan_normal.extend([np.max(element_logits)])
            gt_all_scan_normal.extend([np.max(slicelabel[:original_numslices])])

            # c+=1
            # if c>3:
            #     break
    

    #anomaly samples
    c=0
    pred_anomaly = [] #slice-level prediction excluding <slice_start and >slice_end
    gt_anomaly = [] #slice-level groundtruth excluding <slice_start and >slice_end
    pred_all_anomaly = [] #including <slice_start and >slice_end
    gt_all_anomaly = [] 

    pred_scan_anomaly = [] # scan-level prediction = maximum of anomaly scores throughout all slices excluding <slice_start and >slice_end
    gt_scan_anomaly = [] # scan-level groundtruth  excluding <slice_start and >slice_end
    # If one or more than one slice has an anomaly socre more than <thr>, then the scan-level groundtruth  is anomolous. 
    # If all slices have anomaly score less than <thr>, then the scan-level groundtruth is normal.
    pred_all_scan_anomaly = [] # including <slice_start and >slice_end
    gt_all_scan_anomaly = []

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
            pred_anomaly.extend(element_logits.tolist())
            gt_anomaly.extend(slicelabel[slice_start:slice_end+1].tolist())
            numslices.data.cpu().detach().numpy()

            pred_all_anomaly.extend([0]*slice_start + element_logits.tolist() + [0]*(original_numslices - slice_end - 1)) 
            gt_all_anomaly.extend(slicelabel[:original_numslices].tolist())
            savegraph(element_logits, torch.tensor(slicelabel).unsqueeze(0), str(studyid[0]), original_numslices, slice_start, slice_end, os.path.join(save_path, 'result'))

            pred_scan_anomaly.extend([np.max(element_logits)])
            gt_scan_anomaly.extend([np.max(slicelabel[slice_start:slice_end+1])])
            pred_all_scan_anomaly.extend([np.max(element_logits)])
            gt_all_scan_anomaly.extend([np.max(slicelabel[:original_numslices])])

            # c+=1
            # if c>3:
            #     break

    #Report of metrics 
    # Slice-level
    # Excluding <slice_start and >slice_end 
    metrics_total = calcmetrics(y_true=gt_normal+gt_anomaly, \
                            y_pred=pred_normal+pred_anomaly, \
                            thr=args.thr_conf) #y_pred needs not to be binarized
    #normal test data only
    metrics_normal = calcmetrics(y_true=gt_normal, \
                                y_pred=pred_normal, \
                                thr=args.thr_conf) 
    #anomaly test data only
    metrics_anomaly = calcmetrics(y_true=gt_anomaly, \
                                 y_pred=pred_anomaly, \
                                 thr=args.thr_conf) 
    
    #Including <slice_start and >slice_end 
    metrics_all_total = calcmetrics(y_true=gt_all_normal+gt_all_anomaly, \
                            y_pred=pred_all_normal+pred_all_anomaly, \
                            thr=args.thr_conf) 
    #normal test data only
    metrics_all_normal = calcmetrics(y_true=gt_all_normal, \
                                y_pred=pred_all_normal, \
                                thr=args.thr_conf) 
    #anomaly test data only
    metrics_all_anomaly = calcmetrics(y_true=gt_all_anomaly, \
                                 y_pred=pred_all_anomaly, \
                                 thr=args.thr_conf) 

    # Scan-level
    # Excluding <slice_start and >slice_end 
    metrics_scan_total = calcmetrics(y_true=gt_scan_normal+gt_scan_anomaly, \
                            y_pred=pred_scan_normal+pred_scan_anomaly, \
                            thr=args.thr_conf) #y_pred needs not to be binarized
    #normal test data only
    metrics_scan_normal = calcmetrics(y_true=gt_scan_normal, \
                                y_pred=pred_scan_normal, \
                                thr=args.thr_conf) 
    #anomaly test data only
    metrics_scan_anomaly = calcmetrics(y_true=gt_scan_anomaly, \
                                 y_pred=pred_scan_anomaly, \
                                 thr=args.thr_conf) 
    
    #Including <slice_start and >slice_end 
    metrics_all_scan_total = calcmetrics(y_true=gt_all_scan_normal+gt_all_scan_anomaly, \
                            y_pred=pred_all_scan_normal+pred_all_scan_anomaly, \
                            thr=args.thr_conf) 
    #normal test data only
    metrics_all_scan_normal = calcmetrics(y_true=gt_all_scan_normal, \
                                y_pred=pred_all_scan_normal, \
                                thr=args.thr_conf) 
    #anomaly test data only
    metrics_all_scan_anomaly = calcmetrics(y_true=gt_all_scan_anomaly, \
                                 y_pred=pred_all_scan_anomaly, \
                                 thr=args.thr_conf) 

    df = pd.DataFrame(np.array([metrics_total, metrics_normal, metrics_anomaly, \
                                metrics_all_total, metrics_all_normal, metrics_all_anomaly, \
                                metrics_scan_total, metrics_scan_normal, metrics_scan_anomaly, \
                                metrics_all_scan_total, metrics_all_scan_normal, metrics_all_scan_anomaly, \
                                ]), \
                            columns=['Confusion Matrix', 'AUC', 'Sensitivity (Recall)', 'Specificity', 'Precision', 'F1', 'Accuracy'], \
                            index= ['All scans excluding head/tail slices', 'Normal scans excluding head/tail slices', 'Anomaly scans excluding head/tail slices', \
                                    'All scans including head/tail slices', 'Normal scans including head/tail slices', 'Anomaly scans including head/tail slices', \
                                    'All scans excluding head/tail slices [scan-level]', 'Normal scans excluding head/tail slices [scan-level]', 'Anomaly scans excluding head/tail slices [scan-level]', \
                                    'All scans including head/tail slices [scan-level]', 'Normal scans including head/tail slices [scan-level]', 'Anomaly scans including head/tail slices [scan-level]', \
                                    ])
    df.to_csv(os.path.join(save_path, 'results.csv'))

def test_no_slicelabel(test_normal_loader, test_anomaly_loader, model, device, save_path, args):
    model.eval()
    # if not os.path.exists(os.path.join(save_path, 'result_epoch200')): #fix
    #     os.makedirs(os.path.join(save_path, 'result_epoch200')) #fix
    if not os.path.exists(os.path.join(save_path, 'result')): 
        os.makedirs(os.path.join(save_path, 'result')) 
    #normal samples
    c=0
    # If one or more than one slice has an anomaly socre more than <thr>, then the scan-level groundtruth  is anomolous. 
    # If all slices have anomaly score less than <thr>, then the scan-level groundtruth is normal.
    
    gt_scan_normal = [] # scan-level groundtruth  
    pred_normal = [] #slice-level prediction excluding <slice_start and >slice_end
    pred_all_normal = [] #including <slice_start and >slice_end
    pred_scan_normal = [] # scan-level prediction = maximum of anomaly scores throughout all slices excluding <slice_start and >slice_end
    pred_all_scan_normal = [] # including <slice_start and >slice_end
    

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
            
            pred_normal.extend(element_logits.tolist())
            numslices.data.cpu().detach().numpy()

            pred_all_normal.extend([0]*slice_start + element_logits.tolist() + [0]*(original_numslices - slice_end)) 
            #savegraph(element_logits, slicelabel, str(studyid[0]), original_numslices, slice_start, slice_end, os.path.join(save_path, 'result_epoch200')) #fix
            savegraph(element_logits, slicelabel, str(studyid[0]), original_numslices, slice_start, slice_end, os.path.join(save_path, 'result')) 

            pred_scan_normal.extend([np.max(element_logits)])
            gt_scan_normal.extend([0.0]) #normal label = 1
            pred_all_scan_normal.extend([np.max(element_logits)])
            
            # c+=1
            # if c>3:
            #     break
    

    #anomaly samples
    c=0
    gt_scan_anomaly = [] # scan-level groundtruth
    pred_anomaly = [] #slice-level prediction excluding <slice_start and >slice_end
    pred_all_anomaly = [] #including <slice_start and >slice_end
    pred_scan_anomaly = [] # scan-level prediction = maximum of anomaly scores throughout all slices excluding <slice_start and >slice_end
    pred_all_scan_anomaly = [] # including <slice_start and >slice_end


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
            
            pred_anomaly.extend(element_logits.tolist())
            numslices.data.cpu().detach().numpy()

            pred_all_anomaly.extend([0]*slice_start + element_logits.tolist() + [0]*(original_numslices - slice_end)) 
            #savegraph(element_logits, slicelabel, str(studyid[0]), original_numslices, slice_start, slice_end, os.path.join(save_path, 'result_epoch200')) #fix
            savegraph(element_logits, slicelabel, str(studyid[0]), original_numslices, slice_start, slice_end, os.path.join(save_path, 'result')) 


            pred_scan_anomaly.extend([np.max(element_logits)])
            gt_scan_anomaly.extend([1.0]) #anomaly label = 1
            pred_all_scan_anomaly.extend([np.max(element_logits)])
 
            # c+=1
            # if c>3:
            #     break

    #Report of metrics 
    # Scan-level
    # Excluding <slice_start and >slice_end 
    metrics_scan_total = calcmetrics(y_true=gt_scan_normal+gt_scan_anomaly, \
                            y_pred=pred_scan_normal+pred_scan_anomaly, \
                            thr=args.thr_conf) #y_pred needs not to be binarized
    #normal test data only
    metrics_scan_normal = calcmetrics(y_true=gt_scan_normal, \
                                y_pred=pred_scan_normal, \
                                thr=args.thr_conf) 
    #anomaly test data only
    metrics_scan_anomaly = calcmetrics(y_true=gt_scan_anomaly, \
                                 y_pred=pred_scan_anomaly, \
                                 thr=args.thr_conf) 
    
    #Including <slice_start and >slice_end 
    metrics_all_scan_total = calcmetrics(y_true=gt_scan_normal+gt_scan_anomaly, \
                            y_pred=pred_all_scan_normal+pred_all_scan_anomaly, \
                            thr=args.thr_conf) 
    #normal test data only
    metrics_all_scan_normal = calcmetrics(y_true=gt_scan_normal, \
                                y_pred=pred_all_scan_normal, \
                                thr=args.thr_conf) 
    #anomaly test data only
    metrics_all_scan_anomaly = calcmetrics(y_true=gt_scan_anomaly, \
                                 y_pred=pred_all_scan_anomaly, \
                                 thr=args.thr_conf) 

    df = pd.DataFrame(np.array([metrics_scan_total, metrics_scan_normal, metrics_scan_anomaly, \
                                metrics_all_scan_total, metrics_all_scan_normal, metrics_all_scan_anomaly, \
                                ]), \
                            columns=['Confusion Matrix', 'AUC', 'Sensitivity (Recall)', 'Specificity', 'Precision', 'F1', 'Accuracy'], \
                            index= ['All scans excluding head/tail slices [scan-level]', 'Normal scans excluding head/tail slices [scan-level]', 'Anomaly scans excluding head/tail slices [scan-level]', \
                                    'All scans including head/tail slices [scan-level]', 'Normal scans including head/tail slices [scan-level]', 'Anomaly scans including head/tail slices [scan-level]', \
                                    ])
    #df.to_csv(os.path.join(save_path, 'result_epoch200.csv')) #fix
    df.to_csv(os.path.join(save_path, 'result.csv'))


def eval_ichsubtype(test_normal_loader, test_anomaly_loader, model, device, save_path, args): #evalating subtypes of ICH at slice-level         
    model.eval()
    if not os.path.exists(os.path.join(save_path, 'result')):
        os.makedirs(os.path.join(save_path, 'result'))
    df = pd.read_csv(args.slicelabel_test)  
    pred = [] #slice-level prediction excluding <slice_start and >slice_end
    gt_epidural = []
    gt_intraparenchymal = []
    gt_intraventricular = []
    gt_subarachnoid = []
    gt_subdural = []

    #normal samples
    with torch.no_grad():
        for data in tqdm(test_normal_loader, desc='test_normal', total=len(test_normal_loader)):
            features, scanlabels, slicelabel , studyid , original_numslices, slice_start, slice_end = read_dataset(data)
            numslices = slice_end - slice_start + 1
            maxnumslice = max(numslices)
            features = features[:, :maxnumslice, :].float().to(device)
            if args.model_name == 'model_lstm':
                _, element_logits = model(features, maxnumslice, is_training=False)
            else:
                _, element_logits = model(features, is_training=False)
            element_logits = element_logits.cpu().data.numpy().reshape(-1)[:]

            pred.extend(element_logits.tolist())
            gt_epidural.extend([0]*len(element_logits))
            gt_intraparenchymal.extend([0]*len(element_logits))  
            gt_intraventricular.extend([0]*len(element_logits))  
            gt_subarachnoid.extend([0]*len(element_logits))  
            gt_subdural.extend([0]*len(element_logits))  
            
    #anomaly samples
    with torch.no_grad():
        for data in tqdm(test_anomaly_loader, desc='test_anomaly', total=len(test_anomaly_loader)):
            features, scanlabels, slicelabel , studyid , original_numslices, slice_start, slice_end = read_dataset(data)
            numslices = slice_end - slice_start + 1
            maxnumslice = max(numslices)
            features = features[:, :maxnumslice, :].float().to(device)
            if args.model_name == 'model_lstm':
                _, element_logits = model(features, maxnumslice, is_training=False)
            else:
                _, element_logits = model(features, is_training=False)
            element_logits = element_logits.cpu().data.numpy().reshape(-1)[:]
            pred.extend(element_logits.tolist())
            for numslice in range(numslices):
                pngfilename = studyid[0]+'_'+str(numslice).zfill(3)+'.png'
                idx = df.query('PngFileName=='+'"'+pngfilename+'"').index[0]
                slicelabel_df = df.loc[idx, ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
                
                if slicelabel_df['epidural'] == 1:
                    gt_epidural.append(1)  
                else:
                    gt_epidural.append(0)  
                if slicelabel_df['intraparenchymal'] == 1:
                    gt_intraparenchymal.append(1)
                else:
                    gt_intraparenchymal.append(0)
                if slicelabel_df['intraventricular'] == 1:
                    gt_intraventricular.append(1)
                else:
                    gt_intraventricular.append(0)
                if slicelabel_df['subarachnoid'] == 1:
                    gt_subarachnoid.append(1)
                else:
                    gt_subarachnoid.append(0)
                if slicelabel_df['subdural'] == 1:
                    gt_subdural.append(1)
                else:
                    gt_subdural.append(0)

    metrics_epidural = calcmetrics(y_true=gt_epidural, y_pred=pred, thr=args.thr_conf) #y_pred needs not to be binarized
    metrics_intraparenchymal = calcmetrics(y_true=gt_intraparenchymal, y_pred=pred, thr=args.thr_conf) 
    metrics_intraventricular = calcmetrics(y_true=gt_intraventricular, y_pred=pred, thr=args.thr_conf) 
    metrics_subarachnoid = calcmetrics(y_true=gt_subarachnoid, y_pred=pred, thr=args.thr_conf) 
    metrics_subdural = calcmetrics(y_true=gt_subdural, y_pred=pred, thr=args.thr_conf) 

    df_out = pd.DataFrame(np.array([metrics_epidural, metrics_intraparenchymal, metrics_intraventricular, metrics_subarachnoid, metrics_subdural]), \
                            columns=['Confusion Matrix', 'AUC', 'Sensitivity (Recall)', 'Specificity', 'Precision', 'F1', 'Accuracy'], \
                            index= ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'])
    df_out.to_csv(os.path.join(save_path, 'results_subtype.csv'))
