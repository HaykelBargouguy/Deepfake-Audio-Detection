import pandas as pd
import soundfile as sf
from python_speech_features import logfbank, mfcc
import numpy as np
from CQCC.cqcc import cqcc
import os
from transformers import AutoFeatureExtractor, WavLMForXVector,UniSpeechSatForXVector
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
def compute_feature_sequence_embedding(feature_sequence, mean=True, variance=True, avg_diff=False):
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


# extract cqcc and mfccs
from tqdm import tqdm
import soundfile as sf
import numpy as np


def preprocess_audio_samples(samples, with_compute, mean=True, variance=True, avg_diff=False):
    mfccs = {}
    cqccs = {}

    for key in samples:
        cqccs[key] = []
        mfccs[key] = []

    total_samples = sum(len(samples_list) for samples_list in samples.values())
    progress_bar = tqdm(total=total_samples,bar_format='{l_bar}{bar:20}{r_bar}',desc=f"Preprocessing Audio", unit="samples")

    for key, filepaths in samples.items():
        for filepath in filepaths:
            signal, rate = sf.read(filepath)
            mask = envelope(signal, rate, 0.0005)
            signal = signal[mask]
            feat_cqcc,_,_=extract_cqcc(signal[:2*rate],rate)
            feat_mfcc = compute_mfcc(signal[:2*rate],rate)
            feat_cqcc=pad_length(feat_cqcc,max_len=234)
            feat_mfcc=pad_length(feat_mfcc,max_len=199)
            if with_compute:
                mfccs[key].append(
                    compute_feature_sequence_embedding(feat_mfcc, mean=mean, variance=variance, avg_diff=avg_diff))
                cqccs[key].append(
                    compute_feature_sequence_embedding(feat_cqcc, mean=mean, variance=variance, avg_diff=avg_diff))
            else:
                mfccs[key].append(feat_cqcc.flatten())
                cqccs[key].append(feat_cqcc.flatten())
            progress_bar.update(1)

        mfccs[key] = np.array(mfccs[key])  # Convert to numpy array
        cqccs[key] = np.array(cqccs[key])

    progress_bar.close()  # Close the progress bar after completion
    return mfccs, cqccs

##WavLMForXVector
def wlXvector_embedding(samples):
    feats={}
    for key in samples:
        feats[key]=[]
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
        feats[key]=np.array(feats[key])

    progress_bar.close()
    return feats
def uniSpeechSat_embedding(samples):
    feats={}
    for key in samples:
        feats[key]=[]
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
        feats[key]=np.array(feats[key])

    progress_bar.close()
    return feats

def get_silhouette_score(feats):
    X=np.append(feats["bonafide"],feats["spoof"],axis=0)
    kmeans = KMeans(n_clusters=2, random_state=42)
    silhouette_score(X, kmeans.fit_predict(X))
def compute_evaluation_score(evaluation_feats):
    scores = {}
    computation_times = {}
    total_samples =len(evaluation_feats.keys)
    progress_bar = tqdm(total=total_samples, bar_format='{l_bar}{bar:20}{r_bar}',
                        desc=f"Evaluating Audio Features",
                        unit="Features")
    for feat_name, feats in evaluation_feats.items():
        start_time = time.time()
        scores[feat_name] = get_silhouette_score(feats)
        end_time = time.time()
        computation_times[feat_name] = end_time - start_time
        progress_bar.update(1)
    evaluation_df = pd.DataFrame({
    'Feature Extraction Technique': list(scores.keys()),
    'Silhouette Score': list(scores.values()),
    'Computation Time (seconds)': list(computation_times.values())
})
    progress_bar.close()
    return evaluation_df


def pad_length(feats,max_len):
    num_dim = feats.shape[1]
    if len(feats) > max_len:
        feats = feats[:max_len]
    elif len(feats) < max_len:
        feats = np.concatenate((feats, np.array([[0.] * num_dim] * (max_len - len(feats)))), axis=0)
    return feats
def compute_mfcc(signal, rate, numcep=40, nfilt=60, nfft=1024):
    mel = mfcc(signal[:2 * rate], rate, numcep=numcep, nfilt=nfilt, nfft=nfft)
    return mel


def extract_cqcc(x, fs):
    # INPUT SIGNAL
    x = x.reshape(x.shape[0], 1)  # for one-channel signal
    # PARAMETERS
    B = 96
    fmax = fs / 2
    fmin = fmax / 2 ** 9
    d = 16
    cf = 19
    ZsdD = 'ZsdD'
    # COMPUTE CQCC FEATURES
    CQcc, LogP_absCQT, TimeVec, FreqVec, Ures_LogP_absCQT, Ures_FreqVec, absCQT = cqcc(x, fs, B, fmax, fmin, d, cf,
                                                                                       ZsdD)
    return CQcc, fmax, fmin


# %%
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


def save_embeddings_to_tsv(embeddings, output_dir):
    total_samples = len(embeddings.keys())
    progress_bar = tqdm(total=total_samples, desc=f"Saving tsv Files", unit="files")
    for filename, embedding_data in embeddings.items():
        output_file = os.path.join(output_dir, filename + ".tsv")
        metadata_file = os.path.join(output_dir, filename + "_metadata.tsv")  # Metadata file path

        with open(output_file, 'w') as f, open(metadata_file, 'w') as meta_f:  # Open metadata file

            for label, embedding_samples in embedding_data.items():
                for embedding in embedding_samples:
                    embedding_str = '\t'.join([str(e) for e in embedding])
                    f.write(f"{embedding_str}\n")
                    meta_f.write(f"{label}\n")

        progress_bar.update(1)
    progress_bar.close()



