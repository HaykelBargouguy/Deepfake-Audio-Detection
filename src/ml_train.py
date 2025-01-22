import pickle
import os
import pandas as pd
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from configs.config import ProjectConfig

def train_ml_baselines(data_path):
    cfg = ProjectConfig()
    os.makedirs(cfg.out_fold, exist_ok=True)
    
    print(f"Loading features from {data_path}...")
    data = pd.read_csv(data_path)
    y = data['target'].values
    X = data.drop(['target'], axis=1).values
    
    print('SVM Training Started...')
    svm_clf = SVC(class_weight='balanced', probability=True)
    svm_clf.fit(X, y)
    with open(os.path.join(cfg.out_fold, 'svm_baseline.pkl'), 'wb') as f:
        pickle.dump(svm_clf, f)
    print('SVM Training Completed.')

    print('LightGBM Training Started...')
    lgbm_clf = LGBMClassifier(
        bagging_fraction=0.82, feature_fraction=0.42, 
        max_depth=24, num_leaves=82
    )
    lgbm_clf.fit(X, y)
    with open(os.path.join(cfg.out_fold, 'lightgbm_baseline.pkl'), 'wb') as f:
        pickle.dump(lgbm_clf, f)
    print('LightGBM Training Completed.')

if __name__ == "__main__":
    train_ml_baselines('Data/pa-features/train_feats.csv')