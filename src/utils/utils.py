import os
import pandas as pd
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.manifold import TSNE

def get_label_distribution(label_path):
    """Parses the ASVspoof protocol file and returns the distribution of labels."""
    counts = defaultdict(int)
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                counts[parts[4]] += 1
    return dict(counts)

def plot_tsne(features_dict, output_path, perplexity=30, n_iter=1000):
    """
    Plots 2D t-SNE embeddings for a dictionary of features.
    Args:
        features_dict: {"bonafide": np.array, "spoof": np.array}
        output_path: Filepath to save the resulting PNG plot.
    """
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange']

    for i, (label, feats) in enumerate(features_dict.items()):
        if len(feats) == 0: continue
        
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        embedded = tsne.fit_transform(feats)
        
        plt.scatter(embedded[:, 0], embedded[:, 1], label=label, color=colors[i % len(colors)], alpha=0.6)

    plt.title('t-SNE Spatial Embeddings Visualization')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def create_metadata_df(protocol_path, audio_dir):
    """Creates a structured pandas DataFrame from the ASVspoof protocol text file."""
    df = pd.read_csv(protocol_path, sep=" ", header=None)
    df.columns = ['speaker_id', 'filename', 'null', 'system_id', 'class_name']
    df.drop(columns=['null'], inplace=True)
    df['filepath'] = df['filename'].apply(lambda x: os.path.join(audio_dir, x + '.flac'))
    df['target'] = (df['class_name'] == 'spoof').astype(int)
    return df