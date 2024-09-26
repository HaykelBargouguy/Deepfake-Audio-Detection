import os
import numpy as np
import pickle
import argparse
import pandas as pd
from embedding import extract_mfcc_embeddings , extract_lfcc_embeddings

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str,
                    help='path to data. For example,../Data/PA_CSV/train_data.csv')
parser.add_argument("--type", required=True, type=str, help='type of data: train, dev, or eval')
args = parser.parse_args()

print('Started Preprocessing')

data = pd.read_csv(args.data_path)

files = data['filepath'].values
# Assurer que le chemin est correct
files = [file[3:] for file in files]
feats_mfccs = extract_mfcc_embeddings(files, with_compute=True, mean=True, variance=True, avg_diff=True)
feats_lfccs=extract_lfcc_embeddings(files,with_compute=True,mean=True,variance=True,avg_diff=True)

feats=np.concatenate((feats_lfccs,feats_mfccs),axis=1)

feats = pd.DataFrame(feats)
feats['target'] = data['target'].values
if args.type == 'train':
    feats.to_csv('Data/pa-features/' + args.type+'_'+'feats.csv', index=False)
else:
    feats['system_id'] = data['system_id'].values
    feats.to_csv('Data/pa-features/' + args.type +'_' +'feats.csv', index=False)
print('Ended preprocessing')