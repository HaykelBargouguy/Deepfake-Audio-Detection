import pickle
import argparse
import pandas as pd
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture import BayesianGaussianMixture as DPGMM

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str, help='path to pickled file. For example, data/train.pkl')

args = parser.parse_args()

data = pd.read_csv(args.data_path)
y = data['target'].values
X = data.drop(['target'], axis=1).values

print('SVM training started...')
clf = SVC(class_weight='balanced')
clf.fit(X, y)
print('SVM training completed.')

with open('ML/Models/svm.pkl', 'wb') as outfile:
    pickle.dump(clf, outfile)

# Train GMM for class 0 (bon)
print('GMM training for class 0 (bon) started...')
gmm_bon = GMM(n_components=144, covariance_type='diag', n_init=50)
Xbon = data[data['target'] == 0].drop(['target'], axis=1).values
gmm_bon.fit(Xbon)
print('GMM training for class 0 (bon) completed.')

# Train GMM for class 1 (sp)
print('GMM training for class 1 (sp) started...')
gmm_sp = GMM(n_components=144, covariance_type='diag', n_init=50)
Xsp = data[data['target'] == 1].drop(['target'], axis=1).values
gmm_sp.fit(Xsp)
print('GMM training for class 1 (sp) completed.')

# Train DPGMM for class 0 (bon)
print('DPGMM training for class 0 (bon) started...')
dpgmm_bon = DPGMM(n_components=144, covariance_type='diag', n_init=50, weight_concentration_prior_type='dirichlet_process')
dpgmm_bon.fit(Xbon)
print('DPGMM training for class 0 (bon) completed.')

# Train DPGMM for class 1 (sp)
print('DPGMM training for class 1 (sp) started...')
dpgmm_sp = DPGMM(n_components=144, covariance_type='diag', n_init=50, weight_concentration_prior_type='dirichlet_process')
dpgmm_sp.fit(Xsp)
print('DPGMM training for class 1 (sp) completed.')

# Save models
pickle.dump(gmm_bon, open('ML/Models/bon_gmm.pkl', 'wb'))
pickle.dump(gmm_sp, open('ML/Models/sp_gmm.pkl', 'wb'))
pickle.dump(dpgmm_bon, open('ML/Models/bon_dpgmm.pkl', 'wb'))
pickle.dump(dpgmm_sp, open('ML/Models/sp_dpgmm.pkl', 'wb'))

print('GMM and DPGMM models created')