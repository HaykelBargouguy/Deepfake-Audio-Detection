import librosa
import os
import matplotlib.pyplot as plt
import IPython.display as ipd
import pandas as pd
import soundfile as sf
from collections import defaultdict
from python_speech_features import logfbank, mfcc
import numpy as np
from sklearn.manifold import TSNE
from CQCC.cqcc import cqcc
from transformers import AutoFeatureExtractor, WavLMForXVector
import torch
import ipywidgets as widgets
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

def count_labels(label_path):
    dic_labels = defaultdict(int)  # Initialize as defaultdict with int
    with open(label_path) as file:
        for line in file:
            _, label = line.split()[1], line.split()[-1]
            dic_labels[label] += 1
    return dict(dic_labels)


def plot_label_counts(label_path, title=None):
    label_counts = count_labels(label_path)
    labels = label_counts.keys()
    counts = label_counts.values()

    plt.bar(labels, counts, color=['blue', 'green'])  # Bar plot
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    if title:
        plt.title(title)
    plt.show()


def return_filename2label(label_path):
    filename2label = {}
    num_spoof = 0
    num_bonafide = 0
    for line in open(label_path):
        line = line.split()
        filename, label = line[1], line[-1]
        filename2label[filename] = label
    return filename2label


def create_audio_dataset(metadata_path, audio_data_path, log=False, log_frequency=10):
    processed_files = 0
    df = pd.read_csv(metadata_path, sep=" ", header=None)
    df.columns = ['speaker_id', 'filename', 'null', 'system_id', 'class_name']
    df.drop(columns=['null'], inplace=True)
    df['filepath'] = audio_data_path + df.filename + '.flac'
    df['target'] = (df.class_name == 'spoof').astype('int32')
    total_files = len(os.listdir(audio_data_path))
    lengths = []
    for file_path in df['filepath']:
        sound, sr = sf.read(file_path)
        lengths.append(sound.shape[0] / sr)
        processed_files += 1
        if log and processed_files % (total_files // log_frequency) == 0:
            percentage = (processed_files / total_files) * 100
            print("Progress: {:.2f}%".format(percentage), flush=True)
    df['length'] = lengths
    return df

def create_output_directory(output_dir):
    os.makedirs(output_dir, exist_ok=True)






def plot_and_save_mfcc(diff_mfcc, system_id, output_dir):
    plt.figure(figsize=(10, 4))
    plt.imshow(diff_mfcc, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar()
    plt.title(f'{system_id}')
    plt.grid(False)
    plt.xlabel('Frame')
    plt.ylabel('MFCC Coefficient')
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'differential_mfcc_{system_id}.png')
    plt.savefig(output_path)
    plt.show()


##plot
def get_tsne_embeddings(features, perplexity, iteration, n_components=2):
    embedding = TSNE(n_components=n_components,
                     perplexity=perplexity,
                     n_iter=iteration).fit_transform(features)
    return embedding
def compute_and_plot(mfccs, cqccs, perplexity=50, iteration=5000, n_components=2, output_dir=None):
    # Generate a color map for keys
    color_map = plt.cm.rainbow(np.linspace(0, 1, len(mfccs) + len(cqccs)))
    # Plot MFCCs
    if n_components==3:
        plt.ion()
        ax = plt.axes(projection='3d')

    for i, (key, feature) in enumerate(mfccs.items()):
        embedding = get_tsne_embeddings(feature, perplexity=perplexity, iteration=iteration, n_components=n_components)
        if embedding.shape[1] == 2:
            plt.scatter(embedding[:, 0], embedding[:, 1], label=key, color=color_map[i])
        elif embedding.shape[1] == 3:
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], label=key, color=color_map[i])

    plt.legend()
    plt.title('t-SNE Embeddings for MFCCs')
    plt.figure(figsize=(15, 10)) # Set a larger figure size
    output_path = os.path.join(output_dir, f't-SNE Embeddings for MFCC.png')
    plt.savefig(output_path)
    plt.show()

    # Plot CQCCs
    if n_components==3:
        plt.ion()
        ax = plt.axes(projection='3d')
    for i, (key, feature) in enumerate(cqccs.items()):
        embedding = get_tsne_embeddings(feature, perplexity=perplexity, iteration=iteration, n_components=n_components)
        if embedding.shape[1] == 2:
            plt.scatter(embedding[:, 0], embedding[:, 1], label=key, color=color_map[i])
        elif embedding.shape[1] == 3:
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], label=key, color=color_map[i])

    plt.legend()
    plt.title('t-SNE Embeddings for CQCC')
    plt.figure(figsize=(15, 10))  # Set a larger figure size
    output_path = os.path.join(output_dir, f't-SNE Embeddings for CQCC.png')
    plt.savefig(output_path)
    plt.show()







