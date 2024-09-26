from torch.utils.data import Dataset

import os
import sys


project_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_directory)

from embedding import *
from torch import Tensor
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



"""
For Eval

"""
"""
class SoundDataset(Dataset):
    def __init__(self, data_root, transform, feature_type=None ,is_logical=True, is_train=True, is_eval=False):

        if is_logical:
            track = 'LA'

        else:
            track = 'PA'

        self.feature_type = feature_type
        self.track = track
        self.prefix = 'ASVspoof2019_{}'.format(track)
        self.data_root = data_root
        self.dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
        self.protocols_fname = 'eval.trl' if is_eval else 'train.trn' if is_train else 'dev.trl'
        self.protocols_dir = os.path.join(self.data_root, '{}_cm_protocols/'.format(self.prefix))
        self.files_dir = os.path.join(self.data_root, '{}_{}/'.format(self.prefix, self.dset_name), 'flac/')
        self.protocols_fname = os.path.join(self.protocols_dir,
                                            'ASVspoof2019.{}.cm.{}.txt'.format(track, self.protocols_fname))
        self.transform = transform
        self.sound_files = os.listdir(self.files_dir)
        self.lines = open(self.protocols_fname).readlines()
        self.weights=self.compute_weights()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        sound_file = os.path.join(self.files_dir, self.sound_files[idx])
        # Load the sound file

        tokens = self.lines[idx].strip().split(' ')
        waveform, sample_rate = clean_audio(sound_file)

        waveform = self.transform(waveform)

        if self.feature_type == 'mfcc':

            feats = librosa.feature.mfcc(y=waveform, sr=16000, n_mfcc=40, n_fft=1024)
            delta = librosa.feature.delta(feats)
            delta2 = librosa.feature.delta(delta)
            feats = np.concatenate((feats, delta, delta2), axis=0)

        elif self.feature_type == 'lfcc':

            feats = extract_lfcc(waveform, sample_rate)
        else:
            feats = get_log_spectrum(waveform)

        y = int(tokens[4] == 'spoof')
        feats = Tensor(feats)

        return feats, y

    def compute_weights(self):
        l = len(self.lines)
        sm = 0
        for line in self.lines:
            sm = sm+int(line.strip().split(' ')[4] == 'spoof')
        w = np.array([sm / l, l - sm / l], dtype=np.float32)
        return Tensor(w)

        """

import random 
class SoundDataset(Dataset):
    def __init__(self, data_root, transform, feature_type=None ,is_logical=True, is_train=True, is_eval=False , random_sample = 2 ):

        if is_logical:
            track = 'LA'

        else:
            track = 'PA'

        self.feature_type = feature_type
        self.track = track
        self.prefix = 'ASVspoof2019_{}'.format(track)
        self.data_root = data_root
        self.dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
        self.protocols_fname = 'eval.trl' if is_eval else 'train.trn' if is_train else 'dev.trl'
        self.protocols_dir = os.path.join(self.data_root, '{}_cm_protocols/'.format(self.prefix))
        self.files_dir = os.path.join(self.data_root, '{}_{}/'.format(self.prefix, self.dset_name), 'flac/')
        self.protocols_fname = os.path.join(self.protocols_dir,
                                            'ASVspoof2019.{}.cm.{}.txt'.format(track, self.protocols_fname))
        self.transform = transform
        self.sound_files = os.listdir(self.files_dir)
        self.lines = open(self.protocols_fname).readlines()
        if random_sample: 
            self.sound_files = random.sample(self.sound_files , min(len(self.sound_files), random_sample))
            self.lines = random.sample(self.lines , min(len(self.lines), random_sample))
        self.weights=self.compute_weights()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        sound_file = os.path.join(self.files_dir, self.sound_files[idx])
        # Load the sound file

        tokens = self.lines[idx].strip().split(' ')
        waveform, sample_rate = clean_audio(sound_file)

        waveform = self.transform(waveform)

        if self.feature_type == 'mfcc':

            feats = librosa.feature.mfcc(y=waveform, sr=16000, n_mfcc=40, n_fft=1024)
            delta = librosa.feature.delta(feats)
            delta2 = librosa.feature.delta(delta)
            feats = np.concatenate((feats, delta, delta2), axis=0)

        elif self.feature_type == 'lfcc':

            feats = extract_lfcc(waveform, sample_rate)
        else:
            feats = get_log_spectrum(waveform)

        y = int(tokens[4] == 'spoof')
        feats = Tensor(feats)

        return feats, y

    def compute_weights(self):
        l = len(self.lines)
        sm = 0
        for line in self.lines:
            sm = sm+int(line.strip().split(' ')[4] == 'spoof')
        w = np.array([sm / l, l - sm / l], dtype=np.float32)
        return Tensor(w)

"""
For Dev,Train

"""
"""
import os
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class SoundFeatureDataset(Dataset):
    def __init__(self, data_root, feature_type, is_logical=True, is_train=True, is_eval=False):
        if is_logical:
            track = 'LA'
        else:
            track = 'PA'

        self.track = track
        self.prefix = 'ASVspoof2019_{}'.format(track)
        self.data_root = data_root

        # Determine the dataset name based on flags
        if is_train:
            self.dset_name = 'train'
            self.protocols_fname = 'train.trn'
        elif is_eval:
            self.dset_name = 'eval'
            self.protocols_fname = 'eval.trl'
        else:
            self.dset_name = 'dev'
            self.protocols_fname = 'dev.trl'

        self.protocols_dir = os.path.join(self.data_root, '{}_cm_protocols/'.format(self.prefix))
        self.protocols_fname = os.path.join(self.protocols_dir,
                                            'ASVspoof2019.{}.cm.{}.txt'.format(track, self.protocols_fname))

        # Determine the features save path based on flags
        if is_train:
            self.features_save_path = os.path.join(feature_type, 'train')
        elif is_eval:
            self.features_save_path = os.path.join(feature_type, 'eval')
        else:
            self.features_save_path = os.path.join(feature_type, 'dev')

        self.feature_files = os.listdir(self.features_save_path)
        self.lines = open(self.protocols_fname).readlines()
        self.weights = self.compute_weights()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        feature_file = os.path.join(self.features_save_path, self.feature_files[idx])
        feats = np.load(feature_file)
        tokens = self.lines[idx].strip().split(' ')
        y = int(tokens[4] == 'spoof')
        feats = Tensor(feats)
        return feats, y

    def compute_weights(self):
        l = len(self.lines)
        sm = 0
        for line in self.lines:
            sm += int(line.strip().split(' ')[4] == 'spoof')
        w = np.array([sm / l, (l - sm) / l], dtype=np.float32)
        return Tensor(w)
"""

class SoundFeatureDataset(Dataset):
    def __init__(self,data_root,feature_type,is_logical=True,is_train=True):
        if is_logical:
            track = 'LA'

        else:
            track = 'PA'

        self.track = track
        self.prefix = 'ASVspoof2019_{}'.format(track)
        self.data_root = data_root
        self.dset_name =  'train' if is_train else 'dev'
        self.protocols_fname = 'train.trn' if is_train else 'dev.trl'
        self.protocols_dir = os.path.join(self.data_root, '{}_cm_protocols/'.format(self.prefix))
        self.protocols_fname = os.path.join(self.protocols_dir,
                                            'ASVspoof2019.{}.cm.{}.txt'.format(track, self.protocols_fname))
        if is_train:
            self.features_save_path = os.path.join(feature_type,'train')

        else:
            self.features_save_path = os.path.join(feature_type, 'dev')
        self.feature_files = os.listdir(self.features_save_path)
        self.lines = open(self.protocols_fname).readlines()
        self.weights = self.compute_weights()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        feature_file = os.path.join(self.features_save_path,self.feature_files[idx])
        feats=np.load(feature_file)
        tokens = self.lines[idx].strip().split(' ')
        y = int(tokens[4] == 'spoof')
        feats = Tensor(feats)
        return feats, y

    def compute_weights(self):
        l = len(self.lines)
        sm = 0
        for line in self.lines:
            sm = sm + int(line.strip().split(' ')[4] == 'spoof')
        w = np.array([sm / l, l - sm / l], dtype=np.float32)
        return Tensor(w)
def compute_mfcc_feats(x):
    mfcc = librosa.feature.mfcc(y=x, sr=16000, n_mfcc=24)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(delta)
    feats = np.concatenate((mfcc, delta, delta2), axis=0)
    return feats
def get_log_spectrum(x):
    s = librosa.core.stft(x, n_fft=2048, win_length=2048, hop_length=512)
    a = np.abs(s)**2
    #melspect = librosa.feature.melspectrogram(S=a)
    feat = librosa.power_to_db(a)
    return feat
