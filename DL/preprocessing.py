import argparse
import pandas as pd
import sys
import os
import librosa
# This will add the parent directory to the path, assuming preprocessing.py is in PFA/DL
project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_directory)

from embedding import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", required=True, type=str,
                    help='path to data . For example,../Data/LA_CSV/train.csv')
                    
parser.add_argument("--feature_type", required=True, type=str, help='mfcc,lfcc,spectogram')
parser.add_argument("--type", required=True, type=str, help='train,dev,eval')
args = parser.parse_args()


data = pd.read_csv(args.data_path)
if not os.path.exists(args.feature_type):
    os.mkdir(args.feature_type)

features_save_path = os.path.join(args.feature_type,args.type)
if not os.path.exists(features_save_path):
    os.mkdir(features_save_path)

files = data['filepath'].values

def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = (max_len / x_len) + 1
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x
def get_log_spectrum(x):
    s = librosa.core.stft(x, n_fft=2048, win_length=2048, hop_length=512)
    a = np.abs(s) ** 2
    feat = librosa.power_to_db(a)
    return feat

total_samples = len(files)

print('Started Preprocessing')

progress_bar = tqdm(total=total_samples, bar_format='{l_bar}{bar:20}{r_bar}', desc=f"Preprocessing Audio",
                        unit="samples")


for file in files:
    sig, rate = clean_audio(file)
    sig=pad(sig)
    sig=librosa.util.normalize(sig)
    if args.feature_type == 'mfcc':
        feats = librosa.feature.mfcc(y=sig, sr=16000, n_mfcc=40, n_fft=1024)
        delta = librosa.feature.delta(feats)
        delta2 = librosa.feature.delta(delta)
        feats = np.concatenate((feats, delta, delta2), axis=0)

    elif args.feature_type == 'lfcc':
        feats = extract_lfcc(sig,rate)
    elif args.feature_type =='spectogram':
        feats=get_log_spectrum(sig)
        # print(feats.shape)
        # break
    else:
        raise ValueError('not included')

    progress_bar.update(1)

    filename = os.path.join(features_save_path, os.path.splitext(os.path.basename(file))[0] + '.npy')
    np.save(filename, feats)
progress_bar.close()

print('Ended Preprocessing')

