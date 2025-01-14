import librosa
import numpy as np

def extract_mfcc(waveform, sr=16000, n_mfcc=24):
    """Extracts MFCC with Delta and Delta-Delta."""
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(delta)
    feats = np.concatenate((mfcc, delta, delta2), axis=0)
    return feats

def extract_log_spectrum(waveform, sr=16000):
    """Extracts log power magnitude spectrum."""
    s = librosa.core.stft(waveform, n_fft=2048, win_length=2048, hop_length=512)
    a = np.abs(s)**2
    feat = librosa.power_to_db(a)
    return feat

def get_transform(feature_type):
    """Returns a lambda transform based on desired feature type."""
    if feature_type == 'mfcc':
        return lambda x: extract_mfcc(x)
    elif feature_type == 'spect':
        return lambda x: extract_log_spectrum(x)
    else:
        raise ValueError("Unsupported feature type for direct extraction")