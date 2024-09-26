import os
import json
import shutil
from collections import defaultdict

from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from resnet import *
from loss import *

def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]




def initParams():
    args = {
        'num_epochs': 30,
        'batch_size': 64,
        'lr': 0.0003,
        'lr_decay': 0.5,
        'interval': 10,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'eps': 1e-8,
        'gpu': "1",
        'num_workers': 0,
        'seed': 598,
        'add_loss': "ocsoftmax",
        'weight_loss': 1,
        'r_real': 0.9,
        'r_fake': 0.2,
        'alpha': 20,
        'continue_training': False,
        'out_fold': '../DL/models/resnet-lfcc',
        'enc_dim': 256
    }

    # Change this to specify GPU

    if args['continue_training']:
        assert os.path.exists(args['out_fold'])
    else:
        if not os.path.exists(args['out_fold']):
            os.makedirs(args['out_fold'])
        else:
            shutil.rmtree(args['out_fold'])
            os.mkdir(args['out_fold'])

        if not os.path.exists(os.path.join(args['out_fold'], 'checkpoint')):
            os.makedirs(os.path.join(args['out_fold'], 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args['out_fold'], 'checkpoint'))
            os.mkdir(os.path.join(args['out_fold'], 'checkpoint'))



    args['cuda'] = torch.cuda.is_available()
    print(args['cuda'])
    print(torch.cuda.current_device())
    args['device'] = torch.device("cuda" if args['cuda'] else "cpu")

    return args


def adjust_learning_rate(args, optimizer, epoch_num):
    lr = args['lr'] * (args['lr_decay'] ** (epoch_num // args['interval']))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(args,dev_set, train_set , model):
    lfcc_model = model 
    lfcc_model = lfcc_model.to(args['device'])
    if args['continue_training']:
        lfcc_model = torch.load(os.path.join(args['out_fold'], 'anti-spoofing_lfcc_model.pt')).to(args['device'])

    lfcc_optimizer = torch.optim.Adam(lfcc_model.parameters(), lr=args['lr'],
                                      betas=(args['beta_1'], args['beta_2']), eps=args['eps'], weight_decay=0.0005)

    trainDataLoader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True)

    valDataLoader = DataLoader(dev_set, batch_size=args['batch_size'], shuffle=True)

    criterion = nn.CrossEntropyLoss()

    if args['add_loss'] == "amsoftmax":
        amsoftmax_loss = AMSoftmax(2, args['enc_dim'], s=args['alpha'], m=args['r_real']).to(args['device'])
        amsoftmax_loss.train()
        amsoftmax_optimzer = torch.optim.SGD(amsoftmax_loss.parameters(), lr=0.01)

    if args['add_loss'] == "ocsoftmax":
        ocsoftmax = OCSoftmax(args['enc_dim'], r_real=args['r_real'], r_fake=args['r_fake'], alpha=args['alpha']).to(
            args['device'])
        ocsoftmax.train()
        ocsoftmax_optimzer = torch.optim.SGD(ocsoftmax.parameters(), lr=args['lr'])

    early_stop_cnt = 0
    prev_eer = 1e8

    monitor_loss = args['add_loss']

    for epoch_num in range(args['num_epochs']):
        lfcc_model.train()
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        adjust_learning_rate(args, lfcc_optimizer, epoch_num)
        if args['add_loss'] == "ocsoftmax":
            adjust_learning_rate(args, ocsoftmax_optimzer, epoch_num)
        elif args['add_loss'] == "amsoftmax":
            adjust_learning_rate(args, amsoftmax_optimzer, epoch_num)
        print('\nEpoch: %d ' % (epoch_num + 1))
        for i, (lfcc, labels) in enumerate(tqdm(trainDataLoader)):
            lfcc = lfcc.unsqueeze(1).float().to(args['device']) #for resnet
            lfcc = lfcc.float().to(args['device'])
            labels = labels.type(torch.LongTensor)
            labels = labels.to(args['device'])
            feats, lfcc_outputs = lfcc_model(lfcc)
            lfcc_loss = criterion(lfcc_outputs, labels)

            if args['add_loss'] == "softmax":
                lfcc_optimizer.zero_grad()
                trainlossDict[args['add_loss']].append(lfcc_loss.item())
                lfcc_loss.backward()
                lfcc_optimizer.step()

            if args['add_loss'] == "ocsoftmax":
                ocsoftmaxloss, _ = ocsoftmax(feats, labels)
                lfcc_loss = ocsoftmaxloss * args['weight_loss']
                lfcc_optimizer.zero_grad()
                ocsoftmax_optimzer.zero_grad()
                trainlossDict[args['add_loss']].append(ocsoftmaxloss.item())
                lfcc_loss.backward()
                lfcc_optimizer.step()
                ocsoftmax_optimzer.step()

            if args['add_loss'] == "amsoftmax":
                outputs, moutputs = amsoftmax_loss(feats, labels)
                lfcc_loss = criterion(moutputs, labels)
                trainlossDict[args['add_loss']].append(lfcc_loss.item())
                lfcc_optimizer.zero_grad()
                amsoftmax_optimzer.zero_grad()
                lfcc_loss.backward()
                lfcc_optimizer.step()
                amsoftmax_optimzer.step()
        print("Train Loss: {:.5f}".format(np.nanmean(trainlossDict[monitor_loss])))
        lfcc_model.eval()
        with torch.no_grad():
            idx_loader, score_loader = [], []
            for i, (lfcc, labels) in enumerate(valDataLoader):
                lfcc = lfcc.unsqueeze(1).float().to(args['device']) #for resnet
                lfcc = lfcc.float().to(args['device'])
                labels = labels.type(torch.LongTensor)
                labels = labels.to(args['device'])
                feats, lfcc_outputs = lfcc_model(lfcc)

                lfcc_loss = criterion(lfcc_outputs.float(), labels)
                score = F.softmax(lfcc_outputs, dim=1)[:, 0]

                if args['add_loss'] == "softmax":
                    devlossDict["softmax"].append(lfcc_loss.item())
                elif args['add_loss'] == "amsoftmax":
                    outputs, moutputs = amsoftmax_loss(feats, labels)
                    lfcc_loss = criterion(moutputs, labels)
                    score = F.softmax(outputs, dim=1)[:, 0]
                    devlossDict[args['add_loss']].append(lfcc_loss.item())
                elif args['add_loss'] == "ocsoftmax":
                    ocsoftmaxloss, score = ocsoftmax(feats, labels)
                    devlossDict[args['add_loss']].append(ocsoftmaxloss.item())
                idx_loader.append(labels)
                score_loader.append(score)

            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            val_eer = compute_eer(scores[labels == 0], scores[labels == 1])[0]

            print("Val Loss: {:.5f}".format(np.nanmean(devlossDict[monitor_loss])))
            print("Val EER: {:.5f}".format(val_eer))
            

        torch.save(lfcc_model, os.path.join(args['out_fold'], 'checkpoint',
                                            'anti-spoofing_lfcc_model_%d.pt' % (epoch_num + 1)))
        if args['add_loss'] == "ocsoftmax":
            loss_model = ocsoftmax
            torch.save(loss_model, os.path.join(args['out_fold'], 'checkpoint',
                                                'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
        elif args['add_loss'] == "amsoftmax":
            loss_model = amsoftmax_loss
            torch.save(loss_model, os.path.join(args['out_fold'], 'checkpoint',
                                                'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))
        else:
            loss_model = None

        if val_eer < prev_eer:
            torch.save(lfcc_model, os.path.join(args['out_fold'], 'anti-spoofing_lfcc_model.pt'))
            if args['add_loss'] == "ocsoftmax":
                loss_model = ocsoftmax
                torch.save(loss_model, os.path.join(args['out_fold'], 'anti-spoofing_loss_model.pt'))
            elif args['add_loss'] == "amsoftmax":
                loss_model = amsoftmax_loss
                torch.save(loss_model, os.path.join(args['out_fold'], 'anti-spoofing_loss_model.pt'))
            else:
                loss_model = None
            prev_eer = val_eer
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 100:
            with open(os.path.join(args['out_fold'], 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 19))
            break

    return lfcc_model, loss_model