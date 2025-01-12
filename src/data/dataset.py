import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

def pad_audio(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = (max_len // x_len) + 1
    x_repeat = np.repeat(x, num_repeats)
    return x_repeat[:max_len]

class ASVSpoofDataset(Dataset):
    def __init__(self, data_root, track='LA', is_train=True, is_eval=False, transform=None):
        self.data_root = data_root
        self.track = track
        self.prefix = f'ASVspoof2019_{track}'
        self.transform = transform
        
        if is_eval:
            self.dset_name = 'eval'
            self.protocols_fname = 'eval.trl'
        elif is_train:
            self.dset_name = 'train'
            self.protocols_fname = 'train.trn'
        else:
            self.dset_name = 'dev'
            self.protocols_fname = 'dev.trl'
            
        self.protocols_dir = os.path.join(self.data_root, f'{self.prefix}_cm_protocols/')
        self.files_dir = os.path.join(self.data_root, f'{self.prefix}_{self.dset_name}', 'flac/')
        self.protocols_path = os.path.join(self.protocols_dir, f'ASVspoof2019.{track}.cm.{self.protocols_fname}.txt')
        
        self.sound_files = os.listdir(self.files_dir)
        with open(self.protocols_path, 'r') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        tokens = self.lines[idx].strip().split(' ')
        filename = tokens[1] + '.flac' if self.dset_name != 'eval' else tokens[0] + '.flac'
        sound_file = os.path.join(self.files_dir, filename)
        
        waveform, sr = librosa.load(sound_file, sr=16000)
        waveform = pad_audio(waveform)
        
        if self.transform:
            waveform = self.transform(waveform)
            
        y = int(tokens[4] == 'spoof') if len(tokens) > 4 else 0
        return torch.Tensor(waveform), y