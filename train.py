import torch
import numpy as np
import os
import pathlib
from tqdm import tqdm
from losses import KMXMILL_individual, normal_smooth
from utils import read_dataset

    

def getloss(element_logits, numslices, scanlabels, device, param_k, weights):
    top_k_ce_loss = KMXMILL_individual(element_logits=element_logits,
                                        seq_len=numslices,
                                        labels=scanlabels,
                                        device=device,
                                        param_k=param_k,
                                        loss_type='CE')
    center_loss = normal_smooth(element_logits=element_logits,
                                   labels=scanlabels,
                                   device=device)
    total_loss = float(weights[0]) * top_k_ce_loss + float(weights[1]) * center_loss

    return [top_k_ce_loss, center_loss, total_loss]

def getvalloss(val_loader, model, device, args):
    model.eval()
    with torch.no_grad():
        top_k_ce_loss_val = np.zeros(1, dtype=np.float32)
        center_loss_val= np.zeros(1, dtype=np.float32)
        
        with tqdm(val_loader, desc='val', leave=False) as loader:
            for data in loader:
                features, scanlabels, _ , _ , _, slice_start, slice_end = read_dataset(data)
                scanlabels = scanlabels[:,None].float().to(device)
                numslices = slice_end - slice_start + 1
                maxnumslice = max(numslices)
                numslices = numslices.to(device)
                features = features[:, :maxnumslice, :].float().to(device)
                if args.model_name == 'model_lstm':
                    _, element_logits = model(features, maxnumslice, is_training=False)
                else:
                    _, element_logits = model(features, is_training=False)
                
                weights = args.Lambda.split('_')
                param_k = torch.tensor(args.k).to(device)
                loss = getloss(element_logits, numslices, scanlabels, device, param_k, weights)
                top_k_ce_loss_val += loss[0].data.cpu().detach().numpy()
                center_loss_val += loss[1].data.cpu().detach().numpy()

    return [len(val_loader), top_k_ce_loss_val/len(val_loader), center_loss_val/len(val_loader)]

def validate(epoch, val_normal_loader, val_anomaly_loader, model, device, writer, args):
    [len_loader_normal, top_k_ce_loss_val_normal, center_loss_val_normal] = getvalloss(val_normal_loader, model, device, args)
    [len_loader_anomaly, top_k_ce_loss_val_anomaly, _] = getvalloss(val_anomaly_loader, model, device, args) #center loss is only for normal samples
    top_k_ce_loss_val = (top_k_ce_loss_val_normal * len_loader_normal + top_k_ce_loss_val_anomaly * len_loader_anomaly) / (len_loader_normal + len_loader_anomaly)
    weights = args.Lambda.split('_')
    total_loss_val = float(weights[0]) * top_k_ce_loss_val + float(weights[1]) * center_loss_val_normal 

    writer.add_scalar('Top_k_CE_Loss/val_normal', top_k_ce_loss_val_normal, epoch)
    writer.add_scalar('Top_k_CE_Loss/val_anomaly', top_k_ce_loss_val_anomaly, epoch)
    writer.add_scalar('Top_k_CE_Loss/val_total', top_k_ce_loss_val, epoch)
    writer.add_scalar('Center_Loss/val;', center_loss_val_normal, epoch)
    writer.add_scalar('Total_Loss/val', total_loss_val, epoch)

    return total_loss_val

            
def train_validate(epochs, train_normal_loader, train_anomaly_loader, val_normal_loader, val_anomaly_loader, model, optimizer, writer, device, save_path, args):
    if args.pretrained_ckpt is not None:
        checkpoint = torch.load(args.pretrained_ckpt)
        model.load_state_dict(checkpoint)
        print('Weights loaded from {}'.format(args.pretrained_ckpt))
    else:
        print('model is trained from scratch')

    pdir = pathlib.Path(os.path.join(save_path, 'ckpt'))
    minvalloss = np.inf
    with tqdm(total=epochs, desc='training') as pbar:
        for epoch in range(1, epochs+1):
            model.train()
            for data in zip(train_normal_loader, train_anomaly_loader):
                data_normal, data_anomaly = data
                feature_normal, scanlabel_normal, _ , _ , _, slice_start_normal, slice_end_normal = read_dataset(data_normal)
                feature_anomaly, scanlabel_anomaly, _, _, _, slice_start_anomaly, slice_end_anomaly = read_dataset(data_anomaly)
                
                numslices_normal = slice_end_normal - slice_start_normal
                numslices_anomaly = slice_end_anomaly - slice_start_anomaly
                features = torch.cat((feature_normal, feature_anomaly), dim=0) #shape [30+30scans,60slices,2048etc(feature dim)]
                scanlabels = torch.cat((scanlabel_normal[:,None], scanlabel_anomaly[:,None]),dim=0) #shape [30+30scans,1]
                scanlabels = scanlabels.float().to(device)
                numslices = torch.cat((numslices_normal, numslices_anomaly)).to(device)
                maxnumslice = max(torch.cat((numslices_normal, numslices_anomaly)))
                features = features[:, :maxnumslice, :]
                features = features.float().to(device)

                optimizer.zero_grad()
                if args.model_name == 'model_lstm':
                    _, element_logits = model(features, maxnumslice)
                else:
                    _, element_logits = model(features)
                weights = args.Lambda.split('_')
                param_k = torch.tensor(args.k).to(device)
                
                loss = getloss(element_logits, numslices, scanlabels, device, param_k, weights)
                top_k_ce_loss_train = loss[0]
                center_loss_train = loss[1]
                total_loss_train = loss[2]
            
                total_loss_train.backward()
                optimizer.step()
            
            writer.add_scalar('Top_k_CE_Loss/train', top_k_ce_loss_train, epoch)
            writer.add_scalar('Center_Loss/train', center_loss_train, epoch)
            writer.add_scalar('Total_Loss/train', total_loss_train, epoch)
            # Validation
            total_loss_val = validate(epoch, val_normal_loader, val_anomaly_loader, model, device, writer, args)
            #Snapshot
            if epoch % args.snapshot == 0:
                modelinfo = {'epoch': epoch,\
                    'model_state_dict': model.state_dict(),\
                    'optimizer_state_dict': optimizer.state_dict(),\
                    'loss': total_loss_val
                    }
                savefilename = os.path.join(save_path, 'ckpt', 'snapshot', 'epoch{}_{:.6}.pth'.format(str(epoch).zfill(5), str(float(total_loss_val)).replace('.', '_')))
                torch.save(modelinfo, savefilename)
            #Save the best model
            if epoch == 1 or minvalloss > total_loss_val:
                minvalloss = total_loss_val
                modelinfo = {'epoch': epoch,\
                    'model_state_dict': model.state_dict(),\
                    'optimizer_state_dict': optimizer.state_dict(),\
                    'loss': total_loss_val
                    }
                #Remove currnt pth file if any
                for p in pdir.iterdir():
                    if p.is_file():
                        p.unlink()
                savefilename = os.path.join(save_path, 'ckpt', 'epoch{}_{:.6}.pth'.format(str(epoch).zfill(5), str(float(total_loss_val)).replace('.', '_')))
                torch.save(modelinfo, savefilename)
            pbar.update(1)