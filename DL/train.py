from dataset import *
from torchvision import transforms
import os
import json
import shutil
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from torch import nn
from trainer import *
from models import *
import sys
import os

project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_directory)



from embedding import *


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
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


# feature_fn = compute_mfcc_feats
# transforms = transforms.Compose([
#     lambda x: pad(x),
#     lambda x: librosa.util.normalize(x),
#     lambda x: feature_fn(x),
#     lambda x: Tensor(x)
# ])

train_set = SoundFeatureDataset('../Data/PA', '../DL/lfcc', is_logical=False, is_train=True)
dev_set = SoundFeatureDataset('../Data/PA', '../DL/lfcc', is_logical=False, is_train=False)
args = initParams()
resnet_model= ResNet(3, args['enc_dim'], resnet_type='18', nclasses=2).to(args['device'])
#ast_model = ASTModelVis(input_tdim=1025, label_dim=2, audioset_pretrain=False, imagenet_pretrain=False, model_size='small224')
model = train(args,  dev_set, train_set, resnet_model )

