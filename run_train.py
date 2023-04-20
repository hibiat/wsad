import datetime
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from model import model_generater
from dataset import dataset
from train import train_validate
import parameters
from utils import checkfeaturesize


if __name__ == '__main__':
    args = parameters.parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:{}".format(args.device))
    torch.cuda.set_device(args.device)
    feature_size = checkfeaturesize(args)
    model = model_generater(model_name=args.model_name, feature_size=feature_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    if args.pretrained_ckpt is not None:
        model.load_state_dict(torch.load(args.pretrained_ckpt))

    trainval_normal_dataset = dataset(path_feature=args.path_feature_trainval, \
                            scanlabel=args.scanlabel_trainval, slicelabel=None, classname='normal', \
                            feature_extracted_model=args.feature_extracted_model, max_slicelen=args.max_slicelen, \
                            brainarea_ratio=args.brainarea_ratio)
    trainval_anomaly_dataset = dataset(path_feature=args.path_feature_trainval, \
                            scanlabel=args.scanlabel_trainval, slicelabel=None, classname='anomaly', \
                            feature_extracted_model=args.feature_extracted_model, max_slicelen=args.max_slicelen, \
                            brainarea_ratio=args.brainarea_ratio)
    
    n_val_normal = int(len(trainval_normal_dataset) * args.valratio)
    n_val_anomaly = int(len(trainval_anomaly_dataset) * args.valratio)
    
    val_normal_dataset = Subset(trainval_normal_dataset, list(range(0, n_val_normal)))
    val_anomaly_dataset = Subset(trainval_anomaly_dataset, list(range(0, n_val_anomaly)))
    train_normal_dataset = Subset(trainval_normal_dataset, list(range(n_val_normal, len(trainval_normal_dataset))))
    train_anomaly_dataset = Subset(trainval_anomaly_dataset, list(range(n_val_anomaly, len(trainval_anomaly_dataset))))

    sample_size_adjust = args.sample_size_adjust
    train_normal_loader = DataLoader(dataset=train_normal_dataset, batch_size=args.sample_size-sample_size_adjust, pin_memory=False, num_workers=4, shuffle=True)
    train_anomaly_loader = DataLoader(dataset=train_anomaly_dataset, batch_size=args.sample_size+sample_size_adjust, pin_memory=False, num_workers=4, shuffle=True)
    val_normal_loader = DataLoader(dataset=val_normal_dataset, batch_size=1, pin_memory=False, num_workers=4, shuffle=False)
    val_anomaly_loader = DataLoader(dataset=val_anomaly_dataset, batch_size=1, pin_memory=False, num_workers=4, shuffle=False)
    
    time = datetime.datetime.now()
    time_format = '{}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(time.year, time.month, time.day, time.hour,time.minute, time.second)
    save_path = os.path.join(args.saveroot, args.model_name, args.feature_extracted_model, 'k_{}_Lambda_{}'.format(args.k, args.Lambda))
    if not os.path.exists(os.path.join(save_path, 'ckpt')):
        os.makedirs(os.path.join(save_path, 'ckpt'))
    if not os.path.exists(os.path.join(save_path, 'ckpt', 'snapshot')):
        os.makedirs(os.path.join(save_path, 'ckpt', 'snapshot'))
    with open(file=os.path.join(save_path, 'parameters.txt'), mode='w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))
    writer = SummaryWriter('runs/' + args.model_name + '_' + args.feature_extracted_model + 'k_{}_Lambda_{}_{}'.format(args.k, args.Lambda, time_format))
    
    train_validate(epochs=args.max_epoch, \
        train_normal_loader=train_normal_loader, train_anomaly_loader=train_anomaly_loader, \
        val_normal_loader=val_normal_loader, val_anomaly_loader=val_anomaly_loader, \
        model=model, optimizer=optimizer, writer=writer, device=device, save_path=save_path,\
        args=args)
