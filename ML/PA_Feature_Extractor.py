import os
import numpy as np
import pickle
import argparse
import pandas as pd
from embedding import extract_mfcc_embeddings, extract_lfcc_embeddings

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to data. For example,../Data/LA_CSV/train_data.csv')
parser.add_argument("--type", required=True, type=str, help='type of data: train, dev, or eval')
args = parser.parse_args()

print('Started Preprocessing')

data = pd.read_csv(args.data_path)

files = data['filepath'].values
files = [file[3:] for file in files]


# Extraction des embeddings MFCC
feats_mfccs = extract_mfcc_embeddings(files, with_compute=True, mean=True, variance=True, avg_diff=True)
feats_mfccs_df = pd.DataFrame(feats_mfccs)
feats_mfccs_df['target'] = data['target'].values
if args.type == 'train':
    feats_mfccs_df.to_csv('Data/pa-features/' + args.type + '_' + 'mfcc_feats.csv', index=False)
else:
    feats_mfccs_df['system_id'] = data['system_id'].values
    feats_mfccs_df.to_csv('Data/pa-features/' + args.type + '_' + 'mfcc_feats.csv', index=False)

# Extraction des embeddings LFCC
feats_lfccs = extract_lfcc_embeddings(files, with_compute=True, mean=True, variance=True, avg_diff=True)
feats_lfccs_df = pd.DataFrame(feats_lfccs)
feats_lfccs_df['target'] = data['target'].values
if args.type == 'train':
    feats_lfccs_df.to_csv('Data/pa-features/' + args.type + '_' + 'lfcc_feats.csv', index=False)
else:
    feats_lfccs_df['system_id'] = data['system_id'].values
    feats_lfccs_df.to_csv('Data/pa-features/' + args.type + '_' + 'lfcc_feats.csv', index=False)

print('Ended preprocessing')