import pickle
import argparse
import pandas as pd
from sklearn import svm
from sklearn.mixture import GaussianMixture as GMM
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to pickled file. For example, data/train.pkl')
args = parser.parse_args()
data = pd.read_csv(args.data_path)
y = data['target'].values
X = data.drop(['target'], axis=1).values
print('SVM training started...')
clf = svm.SVC(gamma='auto')
clf.fit(X, y)
with open(f'models/svm.pkl', 'wb') as outfile:
    pickle.dump(clf, outfile)
print('SVM training completed.')
print('Lightgbm training started...')
params={'bagging_fraction': 0.8246575087497517, 'feature_fraction': 0.4216281284504694,
        'max_depth': 24, 'min_child_weight': 23.005002370494537,
        'min_split_gain': 0.029797675986156076, 'num_leaves': 82}
clf_lightgbm = LGBMClassifier()
clf_lightgbm.fit(X, y)
with open(f'models/lightgbm.pkl', 'wb') as outfile:
    pickle.dump(clf_lightgbm, outfile)
print('Lightgbm training completed...')
# Save models




