import pandas as pd
import soundfile as sf
from python_speech_features import mfcc
import numpy as np
import os
from transformers import AutoFeatureExtractor, WavLMForXVector, UniSpeechSatForXVector
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
from numba import njit
from CQCC.CQT_toolbox_2013.cqt import cqt
import librosa
from numpy import log, exp, infty, zeros_like, vstack, zeros, errstate, finfo, sqrt, floor, tile, concatenate, arange, \
    meshgrid, ceil, linspace
from scipy.interpolate import interpn
from scipy.special import logsumexp
from scipy.signal import lfilter
from scipy.fft import dct
from os.path import exists
from random import sample
import math
from tqdm import tqdm
from spafe.utils.spectral import compute_constant_qtransform
from spafe.utils.preprocessing import pre_emphasis, framing, windowing, zero_handling
from spafe.utils.exceptions import ParameterError, ErrorMsgs
import scipy
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix
import lightgbm as lgb
#from bayes_opt import BayesianOptimization

def linear_filter_banks(nfilts=20,
                        nfft=512,
                        fs=16000,
                        low_freq=None,
                        high_freq=None,
                        scale="constant"):
    """
    Compute linear-filterbanks. The filters are stored in the rows, the columns
    correspond to fft bins.

    Args:
        nfilts    (int) : the number of filters in the filterbank.
                          (Default 20)
        nfft      (int) : the FFT size.
                          (Default is 512)
        fs        (int) : sample rate/ sampling frequency of the signal.
                          (Default 16000 Hz)
        low_freq  (int) : lowest band edge of linear filters.
                          (Default 0 Hz)
        high_freq (int) : highest band edge of linear filters.
                          (Default samplerate/2)
        scale    (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
                          Default is "constant"

    Returns:
        (numpy array) array of size nfilts * (nfft/2 + 1) containing filterbank.
        Each row holds 1 filter.
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks


    # compute points evenly spaced in frequency (points are in Hz)
    linear_points = np.linspace(low_freq, high_freq, nfilts + 2)

    # we use fft bins, so we have to convert from Hz to fft bin number
    bins = np.floor((nfft + 1) * linear_points / fs)
    fbank = np.zeros([nfilts, nfft // 2 + 1])

    # init scaler
    if scale == "descendant" or scale == "constant":
        c = 1
    else:
        c = 0

    # compute amps of fbanks
    for j in range(0, nfilts):
        b0, b1, b2 = bins[j], bins[j + 1], bins[j + 2]

        # compute scaler
        if scale == "descendant":
            c -= 1 / nfilts
            c = c * (c > 0) + 0 * (c < 0)

        elif scale == "ascendant":
            c += 1 / nfilts
            c = c * (c < 1) + 1 * (c > 1)

        # compute fbanks
        fbank[j, int(b0):int(b1)] = c * (np.arange(int(b0), int(b1)) -
                                         int(b0)) / (b1 - b0)
        fbank[j, int(b1):int(b2)] = c * (
            int(b2) - np.arange(int(b1), int(b2))) / (b2 - b1)

    return np.abs(fbank)











def compute_feature_sequence_embedding(feature_sequence, mean=False, variance=False, avg_diff=False):
    concat_features = []

    if mean:
        mean_features = np.mean(feature_sequence, axis=0)
        concat_features.append(mean_features)

    if variance:
        stddev_features = np.std(feature_sequence, axis=0)
        concat_features.append(stddev_features)

    if avg_diff:
        average_difference_features = np.zeros((feature_sequence.shape[1],))
        for i in range(0, len(feature_sequence) - 2, 2):
            average_difference_features += feature_sequence[i] - feature_sequence[i + 1]
        average_difference_features /= (len(feature_sequence) // 2)
        average_difference_features = np.array(average_difference_features)
        concat_features.append(average_difference_features)

    if not (mean or variance or avg_diff):
        print('You have to use mean, variance, or avg_diff!')

    return np.hstack(concat_features)


"""

Clean Audio signals :

"""


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def clean_audio(file_path):
    signal, rate = sf.read(file_path)
    mask = envelope(signal, rate, 0.0005)
    signal = signal[mask]
    signal = signal[: rate]

    return signal, rate


def pad_length(signal, signal_rate, max_len):
    # Calculate the length of the input signal in seconds
    signal_len = len(signal) / signal_rate

    # Calculate the number of samples needed to achieve max_len seconds
    required_samples = int(max_len * signal_rate)

    if signal_len <= max_len:
        # If the signal length is less than or equal to max_len, repeat the signal
        # to achieve max_len seconds
        repeat_factor = int(np.ceil(required_samples / len(signal)))
        processed_signal = np.tile(signal, repeat_factor)
    else:
        # If the signal length is longer than max_len, take only max_len seconds
        processed_signal = signal[:required_samples]

    return processed_signal[:required_samples]
# def pad_length(sig, rate, max_len=2):
#     duration_seconds = len(sig) / rate
#
#     required_length = max_len * rate
#
#     if duration_seconds < max_len:
#         padded_length = required_length - len(sig)
#
#         sig = np.concatenate((sig, np.zeros(padded_length)), axis=0)
#
#     return sig
"""

Feature Extraction :

"""

"""

MFCC 

"""

def extract_mfcc_embeddings(files, with_compute, numcep=40, nfilt=60, nfft=1024, max_len=1,
                            mean=False, variance=False, avg_diff=False):
    mfccs = []

    total_samples = len(files)
    progress_bar = tqdm(total=total_samples, bar_format='{l_bar}{bar:20}{r_bar}', desc=f"Preprocessing Audio Files",
                        unit="samples")

    for file in files:
        sig, rate = clean_audio(file)
        sig = pad_length(sig, rate, max_len=max_len)
        mel = mfcc(sig, rate, numcep=numcep, nfilt=nfilt, nfft=nfft)
        if with_compute:
            mfccs.append(compute_feature_sequence_embedding(mel, mean=mean, variance=variance, avg_diff=avg_diff))
        else:
            mfccs.append(mel.flatten())
        progress_bar.update(1)
    mfccs = np.array(mfccs)  # Convert to numpy array
    progress_bar.close()
    return mfccs


"""

CQCC

"""


def cqccDeltas(x, hlen=2):
    win = list(range(hlen, -hlen - 1, -1))
    norm = 2 * (arange(1, hlen + 1) ** 2).sum()
    xx_1 = tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx) / norm
    return D[:, hlen * 2:]


def extract_cqcc(sig, fs, fmin, fmax, B=12, cf=19, d=16):
    # cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    kl = B * math.log(1 + 1 / d, 2)
    gamma = 228.7 * (2 ** (1 / B) - 2 ** (-1 / B))
    eps = 2.2204e-16
    scoeff = 1

    new_fs = 1 / (fmin * (2 ** (kl / B) - 1))
    ratio = 9.562 / new_fs

    Xcq = cqt(sig[:, None], B, fs, fmin, fmax, 'rasterize', 'full', 'gamma', gamma)
    absCQT = abs(Xcq['c'])

    TimeVec = arange(1, absCQT.shape[1] + 1).reshape(1, -1)
    TimeVec = TimeVec * Xcq['xlen'] / absCQT.shape[1] / fs

    FreqVec = arange(0, absCQT.shape[0]).reshape(1, -1)
    FreqVec = fmin * (2 ** (FreqVec / B))

    LogP_absCQT = log(absCQT ** 2 + eps)

    n_samples = int(ceil(LogP_absCQT.shape[0] * ratio))
    Ures_FreqVec = linspace(FreqVec.min(), FreqVec.max(), n_samples)

    xi, yi = meshgrid(TimeVec[0, :], Ures_FreqVec)
    Ures_LogP_absCQT = interpn(points=(TimeVec[0, :], FreqVec[0, :]), values=LogP_absCQT.T, xi=(xi, yi),
                               method='splinef2d')

    CQcepstrum = dct(Ures_LogP_absCQT, type=2, axis=0, norm='ortho')
    CQcepstrum_temp = CQcepstrum[scoeff - 1:cf + 1, :]
    deltas = cqccDeltas(CQcepstrum_temp.T).T

    CQcc = concatenate([CQcepstrum_temp, deltas, cqccDeltas(deltas.T).T], axis=0)

    return CQcc.T


def extract_cqcc_embeddings(files, with_compute, max_len=1, mean=False, variance=False, avg_diff=False):
    cqccs = []

    total_samples = len(files)
    progress_bar = tqdm(total=total_samples, bar_format='{l_bar}{bar:20}{r_bar}', desc=f"Preprocessing Audio Files",
                        unit="samples")

    for file in files:
        sig, rate = clean_audio(file)
        sig = pad_length(sig, rate, max_len=max_len)
        fmax = rate / 2
        fmin = fmax / 2 ** 9
        feat_cqcc = extract_cqcc(sig, rate, fmin=fmin, fmax=fmax)
        if with_compute:
            cqccs.append(
                compute_feature_sequence_embedding(feat_cqcc.T, mean=mean, variance=variance, avg_diff=avg_diff))
        else:
            cqccs.append(feat_cqcc.flatten())
        progress_bar.update(1)
    cqccs = np.array(cqccs)
    progress_bar.close()
    return cqccs


"""

CQT

"""


def extract_cqt_embeddings(files, with_compute, max_len=1, mean=False, variance=False, avg_diff=False):
    cqts = []

    total_samples = len(files)
    progress_bar = tqdm(total=total_samples, bar_format='{l_bar}{bar:20}{r_bar}', desc=f"Preprocessing Audio",
                        unit="samples")

    for file in files:
        sig, rate = clean_audio(file)
        sig = pad_length(sig, rate, max_len=max_len)
        fmax = rate / 2
        fmin = fmax / 2 ** 9
        feat_cqt = np.abs(librosa.cqt(sig,sr=rate,fmin=fmin,bins_per_octave=96))
        if with_compute:
            cqts.append(
                compute_feature_sequence_embedding(feat_cqt.T, mean=mean, variance=variance, avg_diff=avg_diff))
        else:
            cqts.append(feat_cqt.flatten())
        progress_bar.update(1)
    cqts = np.array(cqts)
    progress_bar.close()
    return cqts


"""

LFCC

"""


def lfcc(sig,
         fs=16000,
         num_ceps=20,
         pre_emph=0,
         pre_emph_coeff=0.97,
         win_len=0.030,
         win_hop=0.015,
         win_type="hamming",
         nfilts=70,
         nfft=1024,
         low_freq=None,
         high_freq=None,
         scale="constant",
         dct_type=2,
         normalize=0):
    """
    Compute the linear-frequency cepstral coefficients (GFCC features) from an audio signal.
    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return.
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        win_type       (float) : window type to apply for the windowing.
                                 Default is "hamming".
        nfilts           (int) : the number of filters in the filterbank.
                                 Default is 40.
        nfft             (int) : number of FFT points.
                                 Default is 512.
        low_freq         (int) : lowest band edge of mel filters (Hz).
                                 Default is 0.
        high_freq        (int) : highest band edge of mel filters (Hz).
                                 Default is samplerate / 2 = 8000.
        scale           (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
                                 Default is "constant".
        dct_type         (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                                 Default is 2.
        use_energy       (int) : overwrite C0 with true log energy
                                 Default is 0.
        lifter           (int) : apply liftering if value > 0.
                                 Default is 22.
        normalize        (int) : apply normalization if 1.
                                 Default is 0.
    Returns:
        (array) : 2d array of LFCC features (num_frames x num_ceps)
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    # -> framing
    frames, frame_length = framing(sig=sig,
                                   fs=fs,
                                   win_len=win_len,
                                   win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames,
                        frame_len=frame_length,
                        win_type=win_type)

    # -> FFT -> |.|
    fourrier_transform = np.fft.rfft(windows, nfft)
    abs_fft_values = np.abs(fourrier_transform) ** 2

    #  -> x linear-fbanks
    linear_fbanks_mat = linear_filter_banks(nfilts=nfilts,
                                            nfft=nfft,
                                            fs=fs,
                                            low_freq=low_freq,
                                            high_freq=high_freq,
                                            scale=scale)
    features = np.dot(abs_fft_values, linear_fbanks_mat.T)

    log_features = np.log10(features + 2.2204e-16)

    #  -> DCT(.)
    lfccs = dct(log_features, type=dct_type, norm='ortho', axis=1)[:, :num_ceps]

    return lfccs


def Deltas(x, width=3):
    hlen = int(floor(width/2))
    win = list(range(hlen, -hlen-1, -1))
    xx_1 = tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx)
    return D[:, hlen*2:]


def extract_lfcc(sig,fs, num_ceps=20, order_deltas=2, low_freq=0, high_freq=4000):
    # put VAD here, if wanted
    lfccs = lfcc(sig=sig,
                 fs=fs,
                 num_ceps=num_ceps,
                 low_freq=low_freq,
                 high_freq=high_freq).T
    if order_deltas > 0:
        feats = list()
        feats.append(lfccs)
        for d in range(order_deltas):
            feats.append(Deltas(feats[-1]))
        lfccs = vstack(feats)
    return lfccs




def extract_lfcc_embeddings(files, with_compute, max_len=1, mean=False, variance=False, avg_diff=False):
    lfccs = []

    total_samples = len(files)
    progress_bar = tqdm(total=total_samples, bar_format='{l_bar}{bar:20}{r_bar}', desc=f"Preprocessing Audio",
                        unit="samples")

    for file in files:
        sig, rate = clean_audio(file)
        sig = pad_length(sig, rate, max_len=max_len)
        feat_lfcc = lfcc(sig=sig, fs=rate)
        if with_compute:
            lfccs.append(
                compute_feature_sequence_embedding(feat_lfcc, mean=mean, variance=variance, avg_diff=avg_diff))
        else:
            lfccs.append(feat_lfcc.flatten())
        progress_bar.update(1)

    progress_bar.close()
    lfccs = np.array(lfccs)
    return lfccs


#


"""

WavLMForXVector


"""


def wlXvector_embedding(samples):
    feats = {}
    for key in samples:
        feats[key] = []
    total_samples = sum(len(samples_list) for samples_list in samples.values())
    progress_bar = tqdm(total=total_samples, bar_format='{l_bar}{bar:20}{r_bar}', desc=f"Preprocessing Audio",
                        unit="samples")
    for key, filepaths in samples.items():
        for filepath in filepaths:
            signal, rate = sf.read(filepath)
            mask = envelope(signal, rate, 0.0005)
            signal = signal[mask]
            feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
            model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
            inputs = feature_extractor(signal, sampling_rate=rate, return_tensors="pt")
            with torch.no_grad():
                embeddings = model(**inputs).embeddings
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu().numpy()
            feats[key].append(embeddings)
            progress_bar.update(1)
        feats[key] = np.array(feats[key])

    progress_bar.close()
    return feats


"""

Unispeech

"""


def uniSpeechSat_embedding(samples):
    feats = {}
    for key in samples:
        feats[key] = []
    total_samples = sum(len(samples_list) for samples_list in samples.values())
    progress_bar = tqdm(total=total_samples, bar_format='{l_bar}{bar:20}{r_bar}', desc=f"Preprocessing Audio",
                        unit="samples")
    for key, filepaths in samples.items():
        for filepath in filepaths:
            signal, rate = sf.read(filepath)
            mask = envelope(signal, rate, 0.0005)
            signal = signal[mask]
            feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-sat-base-plus-sv")
            model = UniSpeechSatForXVector.from_pretrained("microsoft/unispeech-sat-base-plus-sv")
            inputs = feature_extractor(signal, sampling_rate=rate, return_tensors="pt")
            # audio file is decoded on the fly
            with torch.no_grad():
                embeddings = model(**inputs).embeddings

            embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
            feats[key].append(embeddings)
            progress_bar.update(1)
        feats[key] = np.array(feats[key])

    progress_bar.close()
    return feats


def save_embeddings_to_tsv(content, output_dir):
    total_samples = len(content)
    progress_bar = tqdm(total=total_samples, desc="Saving tsv Files", unit="files")

    for filename, samples in content.items():
        output_file = os.path.join(output_dir, filename + ".tsv")

        with open(output_file, 'w') as f:
            for sample in samples:
                if isinstance(sample, np.ndarray):
                    sample_str = '\t'.join([str(e) for e in sample])
                else:
                    sample_str = str(sample)

                f.write(f"{sample_str}\n")

        progress_bar.update(1)

    progress_bar.close()


def get_silhouette_score(path, embeddings, label):
    ev_sil_score = {"Embedding": [], "Silhouette Score": []}
    label = pd.read_csv(path+label+'.tsv', delimiter='\t', header=None, names=['label'])
    total_samples = len(embeddings)
    progress_bar = tqdm(total=total_samples, bar_format='{l_bar}{bar:20}{r_bar}', desc=f"Evaluating Audio Features",
                        unit="Features")
    for embedding in embeddings:
        ev_sil_score["Embedding"].append(embedding)
        file = path + embedding + '.tsv'
        data = pd.read_csv(file, delimiter='\t', header=None)
        ev_sil_score["Silhouette Score"].append(silhouette_score(data.values, label.values))
        progress_bar.update(1)
    progress_bar.close()
    ev_sil_score = pd.DataFrame(ev_sil_score)
    return ev_sil_score


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
    return eer * 100


"""
SVM

"""




def compute_ev_score_svm(path, embeddings, label):
    ev_eer_acc = {"Embedding": [], "Accuracy": [], "Equal Error Rate": [], "Computation Time (seconds)": []}
    total_samples = len(embeddings)
    labels = pd.read_csv(path + label + '.tsv', delimiter='\t', header=None, names=['label'])
    progress_bar = tqdm(total=total_samples, bar_format='{l_bar}{bar:20}{r_bar}', desc=f"Evaluating Audio Features",
                        unit="Features")
    for embedding in embeddings:
        ev_eer_acc["Embedding"].append(embedding)
        file = path + embedding + '.tsv'
        data = pd.read_csv(file, delimiter='\t', header=None)
        X_train, X_test, y_train, y_test = train_test_split(data, labels['label'], test_size=0.3, random_state=42)
        clf = svm.NuSVC(gamma="auto")
        start_time = time.time()
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        y_score = clf.decision_function(X_test)
        compute_time = time.time() - start_time
        ev_eer_acc["Accuracy"].append(accuracy)
        ev_eer_acc['Equal Error Rate'].append(get_eer(y_test, y_score))
        ev_eer_acc['Computation Time (seconds)'].append(compute_time)
        progress_bar.update(1)
    progress_bar.close()
    return ev_eer_acc


"""

Xgboost

"""


def compute_ev_score_xgboost(path, embeddings, label):
    ev_eer_acc = {"Embedding": [], "Accuracy": [], "Equal Error Rate": [], "Computation Time (seconds)": []}
    total_samples = len(embeddings)
    labels = pd.read_csv(path + label + '.tsv', delimiter='\t', header=None, names=['label'])
    progress_bar = tqdm(total=total_samples, bar_format='{l_bar}{bar:20}{r_bar}', desc=f"Evaluating Audio Features",
                        unit="Features")
    for embedding in embeddings:
        ev_eer_acc["Embedding"].append(embedding)
        file = path + embedding + '.tsv'
        data = pd.read_csv(file, delimiter='\t', header=None)
        X_train, X_test, y_train, y_test = train_test_split(data, labels['label'], test_size=0.3, random_state=42)
        clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        start_time = time.time()
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        y_score = clf.predict_proba(X_test)[:, 1]
        compute_time = time.time() - start_time
        ev_eer_acc["Accuracy"].append(accuracy)
        ev_eer_acc['Equal Error Rate'].append(get_eer(y_test, y_score))
        ev_eer_acc['Computation Time (seconds)'].append(compute_time)
        progress_bar.update(1)
    progress_bar.close()
    return ev_eer_acc


"""

LightGbm

"""


def compute_ev_score_lgb(path, embeddings, label):
    ev_eer_acc = {"Embedding": [], "Accuracy": [], "Equal Error Rate": [], "Computation Time (seconds)": []}
    total_samples = len(embeddings)
    labels = pd.read_csv(path+label+'.tsv', delimiter='\t', header=None, names=['label'])
    progress_bar = tqdm(total=total_samples, bar_format='{l_bar}{bar:20}{r_bar}', desc=f"Evaluating Audio Features",
                        unit="Features")
    for embedding in embeddings:
        ev_eer_acc["Embedding"].append(embedding)
        file = path + embedding + '.tsv'
        data = pd.read_csv(file, delimiter='\t', header=None)
        X_train, X_test, y_train, y_test = train_test_split(data, labels['label'], test_size=0.3, random_state=42)
        clf = LGBMClassifier()
        start_time = time.time()
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, clf.predict(X_test))
        y_score = clf.predict_proba(X_test)[:, 1]
        compute_time = time.time() - start_time
        ev_eer_acc["Accuracy"].append(accuracy)
        ev_eer_acc['Equal Error Rate'].append(get_eer(y_test, y_score))
        ev_eer_acc['Computation Time (seconds)'].append(compute_time)
        progress_bar.update(1)
    progress_bar.close()
    ev_eer_acc=pd.DataFrame(ev_eer_acc)
    return ev_eer_acc


def bayesion_opt_lgbm(path, embedding, label, init_iter=3, n_iters=7, random_state=11, seed=101, num_iterations=100):
    # Load labels
    labels = pd.read_csv(path + label + '.tsv', delimiter='\t', header=None, names=['label'])

    # Load data
    file = path + embedding + '.tsv'
    data = pd.read_csv(file, delimiter='\t', header=None)

    # Create LightGBM dataset
    dtrain = lgb.Dataset(data=data, label=labels['label'])

    # Split data for optimization process
    X_train, X_test, y_train, y_test = train_test_split(data, labels['label'], test_size=0.2, random_state=42)

    def lgb_eer_score(preds, dtrain):
        fpr, tpr, thresholds = roc_curve(y_test, preds, pos_label=1)
        frr = 1 - tpr
        eer = np.mean(np.min([fpr, frr], axis=0))*100
        return 'eer', eer, False

    dtrain = lgb.LGBMClassifier()
    dtrain.fit(X_train, y_train)
    preds = dtrain.predict_proba(X_test)[:, 1]
    eer_init = lgb_eer_score(preds, y_test)[1]

    def hyp_lgbm(num_leaves, feature_fraction, bagging_fraction, max_depth, min_split_gain, min_child_weight):
        params = {'num_leaves': int(round(num_leaves)),
                  'feature_fraction': max(min(feature_fraction, 1), 0),
                  'bagging_fraction': max(min(bagging_fraction, 1), 0),
                  'max_depth': int(round(max_depth)),
                  'min_split_gain': min_split_gain,
                  'min_child_weight': min_child_weight,
                  'objective': 'binary',
                  'boosting_type': 'gbdt',
                  'learning_rate': 0.05,
                  'n_estimators': num_iterations,
                  'early_stopping_rounds': 50
                  }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric=lgb_eer_score)
        preds = model.predict_proba(X_test)[:, 1]
        eer = lgb_eer_score(preds, y_test)[1]
        return -eer  # Minimize EER

    pds = {'num_leaves': (80, 100),
           'feature_fraction': (0.1, 0.9),
           'bagging_fraction': (0.8, 1),
           'max_depth': (17, 25),
           'min_split_gain': (0.001, 0.1),
           'min_child_weight': (10, 25)
           }

    # Surrogate model
    optimizer = BayesianOptimization(hyp_lgbm, pds, random_state=random_state)

    # Optimize
    optimizer.maximize(init_points=init_iter, n_iter=n_iters)

    # Get optimized parameters
    best_params = optimizer.max['params']
    print("Optimal Parameters:")
    print(best_params)
    optimal_eer = -optimizer.max['target']
    print(f"Optimal EER: {optimal_eer:.2f}%")