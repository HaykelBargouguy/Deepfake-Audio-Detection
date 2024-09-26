import os
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix

from sklearn.svm import SVC
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to pickled file. For example, data/dev_mfcc.pkl')
parser.add_argument("--type", required=True, type=str, help='type of data: dev or eval')
args = parser.parse_args()

print('Started')  # Print start time for the script

def get_eer(y, y_score):
    """
    Calculate the Equal Error Rate (EER) and plot FAR vs FRR curve.

    Args:
        y: True labels.
        y_score: Predicted scores.
        Key: Identifier for the plot.
        output_dir: Directory to save the plot.
    """
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    frr = 1 - tpr

    eer = None
    min_distance = float('inf')
    for i in range(len(fpr)):
        distance = np.abs(fpr[i] - frr[i])
        if distance < min_distance:
            min_distance = distance
            eer = (fpr[i] + frr[i]) / 2

    return eer*100


# Load data
data = pd.read_csv(args.data_path)

# Define the feature type condition

with open('models/svm.pkl', 'rb') as infile:
    clf = pickle.load(infile)

ev_svm = {}

# Start testing SVM
print('Start testing SVM')

# All
X = data.drop(['system_id', 'target'], axis=1).values
y = data['target'].values
y_score = clf.decision_function(X)
ev_svm['All'] = get_eer(y, y_score)

# By system
unique_system_ids = data['system_id'].unique().tolist()
y_bonafide = data[data['target'] == 0]['target'].values
X_bonafide = data[data['target'] == 0].drop(['system_id', 'target'], axis=1).values
for unique_system_id in unique_system_ids:
    if unique_system_id != '-':
        data_system = data[data['system_id'] == unique_system_id]
        y_spoof = data_system['target'].values
        y = np.concatenate((y_spoof, y_bonafide), axis=0)
        X_spoof = data_system.drop(['system_id', 'target'], axis=1).values
        X = np.concatenate((X_spoof, X_bonafide), axis=0)
        indices = np.random.permutation(len(y))
        X = X[indices]
        y = y[indices]
        y_score = clf.decision_function(X)
        ev_svm[unique_system_id] = get_eer(y, y_score)

# End testing SVM
print('End testing SVM')
df_ev_svm = pd.DataFrame.from_dict(ev_svm, orient='index')
df_ev_svm.index.name = 'SVM'
print('Start testing LightGBM')
# Load LightGBM model
lgb = pickle.load(open('models/lightgbm.pkl', 'rb'))

ev_lgb = {}
X = data.drop(['system_id', 'target'], axis=1).values
y = data['target'].values
y_score = lgb.predict_proba(X)[:, 1]
ev_lgb['All'] = get_eer(y, y_score)

# By system
unique_system_ids = data['system_id'].unique().tolist()
y_bonafide = data[data['target'] == 0]['target'].values
X_bonafide = data[data['target'] == 0].drop(['system_id', 'target'], axis=1).values
for unique_system_id in unique_system_ids:
    if unique_system_id != '-':
        data_system = data[data['system_id'] == unique_system_id]
        y_spoof = data_system['target'].values
        y = np.concatenate((y_spoof, y_bonafide), axis=0)
        X_spoof = data_system.drop(['system_id', 'target'], axis=1).values
        X = np.concatenate((X_spoof, X_bonafide), axis=0)
        indices = np.random.permutation(len(y))
        X = X[indices]
        y = y[indices]
        y_score = lgb.predict_proba(X)[:, 1]
        ev_lgb[unique_system_id] = get_eer(y, y_score)

print('End testing LightGBM')
df_ev_lgb = pd.DataFrame.from_dict(ev_lgb,orient='index')
df_ev_lgb.index.name = 'Lighgbm'
df_concatenated = pd.concat([df_ev_svm, df_ev_lgb], axis=1)

df_concatenated.reset_index(inplace=True)

df_concatenated.to_csv('Data/eer_ev_' + args.type+ '.csv', index=False)
print('Ended')




